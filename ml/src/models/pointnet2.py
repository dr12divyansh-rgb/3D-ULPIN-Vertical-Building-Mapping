"""PointNet++ semantic segmentation baseline (PyTorch, CPU-compatible).

Implements the standard set-abstraction (SA) encoder + feature-propagation
(FP) decoder architecture from Qi et al., "PointNet++: Deep Hierarchical
Feature Learning on Point Sets in a Metric Space" (NeurIPS 2017).

Convention (matches the reference implementation):
  * xyz is always point-first: (B, N, 3)
  * per-point features are always channel-first: (B, C, N)

Interface:

    model = PointNet2Seg(num_classes=6)
    logits = model(points)      # points: (B, N, 3) -> logits: (B, N, num_classes)

Only XYZ is consumed (no fake sensor attributes).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def square_distance(src, dst):
    """(B, N, C) x (B, M, C) -> (B, N, M) squared euclidean distances."""
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.transpose(2, 1))
    dist += torch.sum(src ** 2, dim=-1).view(B, N, 1)
    dist += torch.sum(dst ** 2, dim=-1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """Gather points by index: points (B, N, C), idx (B, S) -> (B, S, C)."""
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = (torch.arange(B, dtype=torch.long, device=device)
                     .view(view_shape).repeat(repeat_shape))
    return points[batch_indices, idx, :]


def farthest_point_sample(xyz, npoint):
    """Farthest point sampling. xyz (B, N, 3) -> idx (B, npoint)."""
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.zeros(B, dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """Ball query. Returns (B, S, nsample) indices into xyz (self-padded if sparse)."""
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long, device=device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


class SetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last = out_channel

    def forward(self, xyz, points):
        # xyz: (B, N, 3); points: (B, C, N) channel-first, or None
        B, N, C = xyz.shape
        new_xyz = index_points(xyz, farthest_point_sample(xyz, self.npoint))  # (B, S, 3)
        idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)       # (B, S, K)
        grouped_xyz = index_points(xyz, idx)                                   # (B, S, K, 3)
        grouped_xyz_norm = grouped_xyz - new_xyz.view(B, self.npoint, 1, 3)
        if points is not None:
            grouped_points = index_points(points.permute(0, 2, 1), idx)        # (B, S, K, C)
            new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # (B, S, K, 3+C)
        else:
            new_points = grouped_xyz_norm
        new_points = new_points.permute(0, 3, 2, 1)  # (B, D, K, S)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))
        new_points = torch.max(new_points, dim=2)[0]  # (B, D_out, S) channel-first
        return new_xyz, new_points


class FeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        # Interpolate coarser features (points2 at xyz2) onto finer points xyz1.
        # xyz1 (B, N, 3); xyz2 (B, S, 3); points1 (B, C1, N) or None; points2 (B, C2, S).
        points2 = points2.permute(0, 2, 1)          # (B, S, C2)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape
        dists = square_distance(xyz1, xyz2)         # (B, N, S)
        dists, idx = dists.sort(dim=-1)
        dists, idx = dists[:, :, :3], idx[:, :, :3]
        dist_recip = 1.0 / (dists + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        interpolated = torch.sum(index_points(points2, idx) * weight.unsqueeze(-1), dim=2)  # (B, N, C2)
        if points1 is not None:
            new_points = torch.cat([points1.permute(0, 2, 1), interpolated], dim=-1)
        else:
            new_points = interpolated
        new_points = new_points.permute(0, 2, 1)    # (B, C, N)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))
        return new_points                            # (B, C_out, N)


class PointNet2Seg(nn.Module):
    """PointNet++ semantic segmentation (2 SA + 2 FP layers).

    input_dim=3  → XYZ only (original behaviour, backward-compatible).
    input_dim=6  → XYZRGB: first 3 dims are geometry, last 3 are colour features.
    """

    def __init__(self, num_classes=6, sa1=None, sa2=None, input_dim=3):
        super().__init__()
        sa1 = sa1 or {"npoint": 512, "radius": 0.2, "nsample": 32, "mlp": [64, 64, 128]}
        sa2 = sa2 or {"npoint": 128, "radius": 0.4, "nsample": 32, "mlp": [128, 128, 256]}
        self.input_dim = input_dim
        # SA1 in_channel: 3 local-xyz offsets (always) + (input_dim-3) extra features.
        # For XYZ: in_channel=3 (points=None path uses only xyz-offsets).
        # For XYZRGB: in_channel=6 (xyz-offsets concatenated with grouped RGB).
        self.sa1 = SetAbstraction(sa1["npoint"], sa1["radius"], sa1["nsample"],
                                  input_dim, sa1["mlp"])
        self.sa2 = SetAbstraction(sa2["npoint"], sa2["radius"], sa2["nsample"],
                                  3 + sa1["mlp"][-1], sa2["mlp"])
        self.fp2 = FeaturePropagation(sa2["mlp"][-1] + sa1["mlp"][-1], [256, 128])
        self.fp1 = FeaturePropagation(128, [128, 128, 128])
        self.conv_out = nn.Conv1d(128, num_classes, 1)

    def forward(self, points):
        """points: (B, N, input_dim) -> logits: (B, N, num_classes).

        First 3 channels are always XYZ coordinates used for geometry (FPS,
        ball-query, FP interpolation).  Channels 3+ are per-point features
        (e.g. RGB) fed into SA1 alongside the local xyz offsets.
        """
        xyz = points[:, :, :3]                                      # (B, N, 3)
        extra = (points[:, :, 3:].permute(0, 2, 1)                  # (B, C_extra, N)
                 if self.input_dim > 3 else None)
        l1_xyz, l1_points = self.sa1(xyz, extra)                    # (B, 512, 128)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)             # (B, 128, 256)
        x2 = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)         # (B, 128, 512)
        x1 = self.fp1(xyz, l1_xyz, None, x2)                        # (B, 128, N)
        logits = self.conv_out(x1)                                   # (B, C, N)
        return logits.permute(0, 2, 1)                               # (B, N, C)
