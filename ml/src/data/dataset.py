"""PyTorch dataset for synthetic LiDAR semantic segmentation.

Design notes
------------
* Only XYZ is used as model input. ``intensity`` / ``return_number`` are
  synthetic placeholders (Step 1/2) and are deliberately NOT used.
* Original coordinates are always preserved (returned as ``xyz_orig``); the
  model sees per-chunk normalised coordinates (``xyz``).
* Training chunks are sampled from training scenes only; validation/test
  scenes are never sampled for training (scene-level split from Step 1/2).
* Chunk sampling is deterministic (seeded) so runs are reproducible.

STPLS3D additions (Step 4)
--------------------------
* ``STPLS3DChunkDataset`` reads preprocessed .npz chunks produced by
  ``ml/src/data/prepare_stpls3d.py``.  Chunks are loaded lazily (one per
  __getitem__ call) so the full 35 GB dataset never lives in RAM.
* ``build_dataset()`` is a factory that dispatches on
  ``cfg["dataset"]["source"]`` (``"synthetic"`` | ``"stpls3d"``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from synthetic_city.core import ply_io
from synthetic_city.core.labels import CLASS_NAMES, CLASS_IDS, NON_BUILDING_ID

SPLITS = ("train", "validation", "test")


def validate_classes(config_classes: dict) -> None:
    """Refuse to run if the config class mapping drifts from the repo's canonical ids."""
    for name, cid in config_classes.items():
        if CLASS_IDS.get(name) != int(cid):
            raise ValueError(
                f"config class mapping mismatch: {name}->{cid} but repo defines "
                f"{name}->{CLASS_IDS.get(name)}. Fix the config or the data."
            )
    if len(config_classes) != len(CLASS_NAMES):
        raise ValueError("config class mapping has a different size than the repo definition")


def list_scene_dirs(data_root, split) -> list[Path]:
    root = Path(data_root) / split
    if not root.exists():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir()])


def load_scene_pointcloud(scene_dir):
    """Return (xyz (N,3) float32, labels (N,) int64, building_id (N,) int64)."""
    data, _ = ply_io.read_pointcloud(Path(scene_dir) / "lidar" / "pointcloud.ply")
    xyz = np.column_stack([
        np.asarray(data["x"], dtype=np.float32),
        np.asarray(data["y"], dtype=np.float32),
        np.asarray(data["z"], dtype=np.float32),
    ])
    labels = np.asarray(data["class_id"], dtype=np.int64)
    building_id = np.asarray(data["building_id"], dtype=np.int64)
    return xyz, labels, building_id


def normalize_chunk(xyz: np.ndarray):
    """Centre to zero mean and scale to a unit sphere. Returns (norm, centroid, scale)."""
    centroid = xyz.mean(axis=0, keepdims=True)
    centered = xyz - centroid
    scale = float(np.max(np.linalg.norm(centered, axis=1)))
    if scale < 1e-8:
        scale = 1.0
    return (centered / scale).astype(np.float32), centroid.reshape(-1), scale


def chunk_indices(n: int, chunk_size: int, seed: int) -> list[np.ndarray]:
    """Deterministically partition [0, n) into covering chunks of ~chunk_size.

    Used to predict on a full scene point cloud that is larger than the
    network's fixed input size. Points are randomly (seeded) permuted then cut
    into contiguous chunks, so every point is predicted exactly once.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    chunks = []
    for start in range(0, n, chunk_size):
        chunks.append(perm[start:start + chunk_size])
    return chunks


def compute_class_weights(data_root, num_classes) -> np.ndarray:
    """Inverse-frequency class weights from the training split (for the CE loss).

    Returns a float32 array of length num_classes, normalised to mean 1.
    Used to counter the strong class imbalance (building dominates) without
    fabricating any label information.
    """
    counts = np.zeros(num_classes, dtype=np.float64)
    for scene_dir in list_scene_dirs(data_root, "train"):
        _, labels, _ = load_scene_pointcloud(scene_dir)
        counts += np.bincount(labels, minlength=num_classes)
    counts = np.maximum(counts, 1.0)  # avoid division by zero for absent classes
    weights = counts.sum() / (num_classes * counts)
    weights /= weights.mean()
    return weights.astype(np.float32)


class SceneChunkDataset(Dataset):
    """Yields fixed-size, deterministically-sampled chunks from a split.

    ``cfg`` is the *full* segmentation config; ``samples_per_scene`` controls
    how many chunks are drawn from each scene.
    """

    def __init__(self, data_root, split, cfg, samples_per_scene, seed):
        validate_classes(cfg["classes"])
        self.num_points = int(cfg["dataset"]["num_points"])
        rng = np.random.default_rng(seed)
        self.scenes = []
        self.chunks = []  # (scene_idx, indices)
        scene_dirs = list_scene_dirs(data_root, split)
        for si, scene_dir in enumerate(scene_dirs):
            xyz, labels, _ = load_scene_pointcloud(scene_dir)
            self.scenes.append((xyz, labels, scene_dir.name))
            n = xyz.shape[0]
            for _ in range(samples_per_scene):
                idx = rng.choice(n, self.num_points, replace=(n < self.num_points))
                self.chunks.append((si, idx))
        if not self.scenes:
            raise RuntimeError(f"no scenes found in {data_root}/{split}")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, i):
        si, idx = self.chunks[i]
        xyz, labels, scene_name = self.scenes[si]
        xyz_chunk = xyz[idx]
        lab_chunk = labels[idx]
        xyz_norm, _, _ = normalize_chunk(xyz_chunk)
        return {
            "xyz": torch.from_numpy(xyz_norm),
            "xyz_orig": torch.from_numpy(xyz_chunk),
            "labels": torch.from_numpy(lab_chunk),
            "scene": scene_name,
        }


# ---------------------------------------------------------------------------
# STPLS3D dataset — reads preprocessed .npz chunks (never loads full 35 GB)
# ---------------------------------------------------------------------------

import json  # noqa: E402  (standard-library, already available)

_STPLS3D_CLASS_NAMES = ["ground", "building", "vegetation", "road", "vehicle", "other"]


class STPLS3DChunkDataset(Dataset):
    """Load pre-chunked STPLS3D data from a manifest + .npz files.

    Produced by ``ml/src/data/prepare_stpls3d.py``.  Each .npz contains:
      * ``xyz``    — float32 (4096, 3)
      * ``labels`` — uint8   (4096,)
      * ``rgb``    — uint8   (4096, 3)

    Only XYZ is fed to the model (mode_a).  RGB is present in each chunk for
    future XYZRGB experiments (mode_b) but ignored in the first experiment.

    Parameters
    ----------
    data_root : str | Path
        Directory that contains ``manifest.json`` and the ``chunks/`` sub-tree.
    split : str
        One of ``"train"``, ``"validation"``, ``"test"``.
    num_points : int
        Expected points per chunk (used only for validation; must match manifest).
    input_mode : str
        ``"xyz"`` (mode A) — model receives normalised XYZ only.
        ``"xyzrgb"`` (mode B) — model receives normalised XYZ + normalised RGB.
    """

    NUM_CLASSES = 6
    CLASS_NAMES = _STPLS3D_CLASS_NAMES

    def __init__(
        self,
        data_root,
        split: str,
        num_points: int = 4096,
        input_mode: str = "xyz",
        preload: bool = False,
    ):
        """
        preload : bool
            If True, load all .npz chunk files into RAM during __init__.
            Eliminates per-step file I/O overhead at the cost of memory.
            At 4096 pts/chunk with xyz+labels+rgb, 1500 chunks ≈ 97 MB.
            Strongly recommended for CPU training on OneDrive-backed paths.
        """
        if split not in ("train", "validation", "test"):
            raise ValueError(f"split must be train/validation/test, got {split!r}")
        if input_mode not in ("xyz", "xyzrgb"):
            raise ValueError(f"input_mode must be 'xyz' or 'xyzrgb', got {input_mode!r}")

        self.data_root = Path(data_root)
        self.split = split
        self.num_points = num_points
        self.input_mode = input_mode

        manifest_path = self.data_root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"manifest.json not found in {self.data_root}. "
                "Run prepare_stpls3d.py first."
            )
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_meta = json.load(f)

        self.entries = [
            e for e in manifest_meta["chunks"] if e["split"] == split
        ]
        if not self.entries:
            raise RuntimeError(
                f"No '{split}' chunks found in manifest {manifest_path}. "
                "Check that preprocessing produced chunks for this split."
            )

        # Optional preload: cache all arrays in RAM to eliminate per-step IO
        self._cache: list[dict] | None = None
        if preload:
            print(f"  [STPLS3D] preloading {len(self.entries)} chunks "
                  f"({split} split) into RAM...", flush=True)
            self._cache = []
            for entry in self.entries:
                npz_path = self.data_root / "chunks" / entry["path"]
                d = np.load(npz_path)
                self._cache.append({
                    "xyz": d["xyz"].astype(np.float32).copy(),
                    "labels": d["labels"].astype(np.int64).copy(),
                    "rgb": (d["rgb"].astype(np.float32) / 255.0).copy(),
                    "scene_id": entry["scene_id"],
                })
            total_mb = sum(
                c["xyz"].nbytes + c["labels"].nbytes + c["rgb"].nbytes
                for c in self._cache
            ) / 1_048_576
            print(f"  [STPLS3D] preloaded {len(self._cache)} chunks "
                  f"({total_mb:.0f} MB)", flush=True)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int) -> dict:
        if self._cache is not None:
            cached = self._cache[i]
            xyz = cached["xyz"]
            labels = cached["labels"]
            rgb = cached["rgb"]
            scene_id = cached["scene_id"]
        else:
            entry = self.entries[i]
            npz_path = self.data_root / "chunks" / entry["path"]
            with np.load(npz_path) as data:
                xyz = data["xyz"].astype(np.float32, copy=False)      # no-copy if stored as float32
                labels = data["labels"].astype(np.int64, copy=False)  # no-copy if stored as int64
                rgb = data["rgb"].astype(np.float32) * (1.0 / 255.0)  # uint8→float32 (unavoidable)
            scene_id = entry["scene_id"]

        if xyz.shape[0] != self.num_points:
            raise RuntimeError(
                f"Chunk index {i} has {xyz.shape[0]} points, "
                f"expected {self.num_points}"
            )

        xyz_norm, _, _ = normalize_chunk(xyz)

        if self.input_mode == "xyzrgb":
            features = np.concatenate([xyz_norm, rgb], axis=1)  # both float32; no cast needed
        else:
            features = xyz_norm  # (N, 3)

        return {
            "xyz": torch.from_numpy(features),        # (N, 3) or (N, 6)
            "xyz_orig": torch.from_numpy(xyz),        # (N, 3) unnormalised
            "labels": torch.from_numpy(labels),       # (N,)
            "scene": scene_id,
        }

    def class_distribution(self) -> dict:
        """Aggregate per-class point counts from manifest (no file I/O)."""
        counts = {c: 0 for c in range(self.NUM_CLASSES)}
        for e in self.entries:
            for c, n in e["per_class"].items():
                counts[int(c)] += n
        return counts

    def summary(self) -> str:
        """Return a human-readable summary string."""
        n_scenes = len({e["scene_id"] for e in self.entries})
        total_pts = sum(e["n_points"] for e in self.entries)
        dist = self.class_distribution()
        total_valid = sum(dist.values()) or 1
        lines = [
            f"STPLS3DChunkDataset  split={self.split}  mode={self.input_mode}",
            f"  chunks : {len(self.entries):,}",
            f"  scenes : {n_scenes}",
            f"  points : {total_pts:,}",
            "  class distribution (after mapping):",
        ]
        for c, name in enumerate(_STPLS3D_CLASS_NAMES):
            cnt = dist[c]
            pct = 100.0 * cnt / total_valid
            lines.append(f"    {c} {name:<12} {cnt:>12,}  ({pct:.1f}%)")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dataset factory — dispatches on cfg["dataset"]["source"]
# ---------------------------------------------------------------------------

def subset_dataset(ds: Dataset, max_items: int, seed: int) -> Dataset:
    """Return a deterministic reproducible Subset of at most max_items samples.

    Uses a seeded permutation so the same subset is used across restarts.
    If len(ds) <= max_items, returns ds unchanged (no wrapping overhead).
    """
    if len(ds) <= max_items:
        return ds
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(ds))[:max_items].tolist()
    return torch.utils.data.Subset(ds, indices)


def _resolve(path_str: str) -> Path:
    """Resolve a config path: absolute if it is absolute, else relative to project root."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    # Project root = 3 levels above this file (ml/src/data/dataset.py → project root)
    project_root = Path(__file__).resolve().parents[3]
    resolved = project_root / p
    return resolved


def build_dataset(cfg: dict, split: str, seed: int, samples_per_scene: int | None = None):
    """Return the appropriate Dataset for the given config and split.

    Recognises:
      cfg["dataset"]["source"] == "synthetic"  (default, backwards-compatible)
      cfg["dataset"]["source"] == "stpls3d"
      cfg["dataset"]["source"] == "mixed"

    For the synthetic dataset ``samples_per_scene`` must be provided (taken from
    the relevant config key by the caller).  For STPLS3D/mixed it is ignored
    because chunks are pre-computed.
    """
    source = cfg["dataset"].get("source", "synthetic")

    if source == "synthetic":
        data_root = _resolve(cfg["dataset"]["data_root"])
        if samples_per_scene is None:
            key = "train_samples_per_scene" if split == "train" else "val_samples_per_scene"
            samples_per_scene = int(cfg["dataset"][key])
        return SceneChunkDataset(
            data_root, split, cfg,
            samples_per_scene=samples_per_scene,
            seed=seed,
        )

    if source == "stpls3d":
        data_root = _resolve(cfg["dataset"]["data_root"])
        num_points = int(cfg["dataset"].get("num_points", 4096))
        input_mode = cfg["dataset"].get("input_mode", "xyz")
        # Apply optional cap BEFORE preloading so we only cache the subset
        max_key = "max_train_chunks" if split == "train" else "max_val_chunks"
        max_n = cfg["dataset"].get(max_key)
        preload = bool(cfg["dataset"].get("preload", False))
        ds = STPLS3DChunkDataset(data_root, split, num_points=num_points,
                                 input_mode=input_mode, preload=False)
        if max_n is not None:
            ds = subset_dataset(ds, int(max_n), seed)
        # If preload requested and we have a Subset, rebuild with preload
        if preload:
            # Preload by extracting the subset indices and building a new dataset
            if isinstance(ds, torch.utils.data.Subset):
                subset_entries = [ds.dataset.entries[i] for i in ds.indices]
                # Build a preloaded mini-dataset
                ds_base = STPLS3DChunkDataset.__new__(STPLS3DChunkDataset)
                ds_base.data_root = data_root
                ds_base.split = split
                ds_base.num_points = num_points
                ds_base.input_mode = input_mode
                ds_base.entries = subset_entries
                ds_base._cache = None
                # Now preload
                print(f"  [STPLS3D] preloading {len(subset_entries)} {split} chunks...", flush=True)
                ds_base._cache = []
                for entry in subset_entries:
                    npz_path = data_root / "chunks" / entry["path"]
                    d = np.load(npz_path)
                    ds_base._cache.append({
                        "xyz": d["xyz"].astype(np.float32).copy(),
                        "labels": d["labels"].astype(np.int64).copy(),
                        "rgb": (d["rgb"].astype(np.float32) / 255.0).copy(),
                        "scene_id": entry["scene_id"],
                    })
                total_mb = sum(c["xyz"].nbytes + c["labels"].nbytes + c["rgb"].nbytes
                               for c in ds_base._cache) / 1_048_576
                print(f"  [STPLS3D] preloaded {len(ds_base._cache)} chunks ({total_mb:.0f} MB)", flush=True)
                ds = ds_base
            else:
                ds = STPLS3DChunkDataset(data_root, split, num_points=num_points,
                                         input_mode=input_mode, preload=True)
        return ds

    if source == "mixed":
        ds = _build_mixed_dataset(cfg, split, seed)
        max_key = "max_train_chunks" if split == "train" else "max_val_chunks"
        max_n = cfg["dataset"].get(max_key)
        if max_n is not None:
            ds = subset_dataset(ds, int(max_n), seed)
        return ds

    raise ValueError(f"Unknown dataset source: {source!r}. Choose 'synthetic', 'stpls3d', or 'mixed'.")


def _build_mixed_dataset(cfg: dict, split: str, seed: int):
    """Build the MixedDataset for source='mixed'."""
    syn_sub = cfg["dataset"]["synthetic"]
    syn_cfg = {"dataset": syn_sub, "classes": cfg["classes"]}
    stpls3d_root = _resolve(cfg["dataset"]["stpls3d_root"])
    syn_root = _resolve(syn_sub["data_root"])
    num_points = int(cfg["dataset"].get("num_points", 4096))
    input_mode = cfg["dataset"].get("input_mode", "xyz")
    synthetic_weight = float(cfg["dataset"].get("synthetic_weight", 0.3))
    preload = bool(cfg["dataset"].get("preload", False))

    syn_key = "train_samples_per_scene" if split == "train" else "val_samples_per_scene"
    samples_per_scene = int(syn_sub.get(syn_key, 24 if split == "train" else 16))
    syn_ds = SceneChunkDataset(
        syn_root, split, syn_cfg,
        samples_per_scene=samples_per_scene, seed=seed,
    )
    # For preloaded mixed training, limit STPLS3D to a manageable subset first.
    # max_stpls3d_chunks (default 1200) controls how many STPLS3D chunks are loaded.
    max_st = cfg["dataset"].get("max_stpls3d_chunks", 1200 if preload else None)
    stpls3d_ds_full = STPLS3DChunkDataset(stpls3d_root, split, num_points=num_points,
                                           input_mode=input_mode, preload=False)
    if max_st is not None and len(stpls3d_ds_full) > int(max_st):
        # Build a pre-selected subset dataset with preload support
        rng_st = np.random.default_rng(seed + 7)  # different seed from main subset
        chosen_idx = rng_st.permutation(len(stpls3d_ds_full))[:int(max_st)]
        if preload:
            print(f"  [STPLS3D/mixed] preloading {len(chosen_idx)} {split} chunks...", flush=True)
            cache = []
            for idx in chosen_idx:
                entry = stpls3d_ds_full.entries[idx]
                npz_path = stpls3d_root / "chunks" / entry["path"]
                d = np.load(npz_path)
                cache.append({
                    "xyz": d["xyz"].astype(np.float32).copy(),
                    "labels": d["labels"].astype(np.int64).copy(),
                    "rgb": (d["rgb"].astype(np.float32) / 255.0).copy(),
                    "scene_id": entry["scene_id"],
                })
            total_mb = sum(c["xyz"].nbytes + c["labels"].nbytes + c["rgb"].nbytes
                           for c in cache) / 1_048_576
            print(f"  [STPLS3D/mixed] preloaded {len(cache)} chunks ({total_mb:.0f} MB)", flush=True)
            # Build a preloaded mini-dataset
            stpls3d_ds = STPLS3DChunkDataset.__new__(STPLS3DChunkDataset)
            stpls3d_ds.data_root = stpls3d_root
            stpls3d_ds.split = split
            stpls3d_ds.num_points = num_points
            stpls3d_ds.input_mode = input_mode
            stpls3d_ds.entries = [stpls3d_ds_full.entries[i] for i in chosen_idx]
            stpls3d_ds._cache = cache
        else:
            stpls3d_ds = torch.utils.data.Subset(stpls3d_ds_full, chosen_idx.tolist())
    else:
        stpls3d_ds = STPLS3DChunkDataset(stpls3d_root, split, num_points=num_points,
                                          input_mode=input_mode, preload=preload)
    return MixedDataset(syn_ds, stpls3d_ds, synthetic_weight=synthetic_weight, seed=seed)


# ---------------------------------------------------------------------------
# Class-weight helpers
# ---------------------------------------------------------------------------

def compute_class_weights_stpls3d(data_root, num_classes: int, split: str = "train") -> np.ndarray:
    """Inverse-frequency class weights computed from the STPLS3D manifest.

    Reads only the manifest.json (no .npz files), so it is fast regardless of
    dataset size.  Returns float32 array of length num_classes, normalised to
    mean 1.
    """
    manifest_path = Path(data_root) / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    counts = np.zeros(num_classes, dtype=np.float64)
    for entry in manifest["chunks"]:
        if entry["split"] != split:
            continue
        for c, n in entry["per_class"].items():
            counts[int(c)] += n

    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (num_classes * counts)
    weights /= weights.mean()
    return weights.astype(np.float32)


def compute_class_weights_mixed(
    syn_data_root, stpls3d_data_root,
    num_classes: int,
    synthetic_weight: float = 0.5,
) -> np.ndarray:
    """Inverse-frequency weights for the mixed dataset.

    Computes counts from both sources, then blends with the given ratio
    (synthetic_weight = fraction of total weight given to synthetic counts).
    """
    # Synthetic counts
    syn_counts = np.zeros(num_classes, dtype=np.float64)
    for scene_dir in list_scene_dirs(syn_data_root, "train"):
        _, labels, _ = load_scene_pointcloud(scene_dir)
        syn_counts += np.bincount(labels, minlength=num_classes)

    # STPLS3D counts (from manifest)
    stpls3d_counts = np.zeros(num_classes, dtype=np.float64)
    manifest_path = Path(stpls3d_data_root) / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    for entry in manifest["chunks"]:
        if entry["split"] != "train":
            continue
        for c, n in entry["per_class"].items():
            stpls3d_counts[int(c)] += n

    # Blend by normalising each to total-count = 1, then weight
    syn_norm = syn_counts / (syn_counts.sum() or 1.0)
    stpls3d_norm = stpls3d_counts / (stpls3d_counts.sum() or 1.0)
    blended = synthetic_weight * syn_norm + (1.0 - synthetic_weight) * stpls3d_norm
    blended = np.maximum(blended, 1e-6)
    weights = 1.0 / (num_classes * blended)
    weights /= weights.mean()
    return weights.astype(np.float32)


# ---------------------------------------------------------------------------
# MixedDataset — combines synthetic SceneChunkDataset + STPLS3DChunkDataset
# ---------------------------------------------------------------------------

class MixedDataset(Dataset):
    """Epoch-balanced mix of synthetic and STPLS3D chunks.

    Strategy: each epoch has ``len(larger_ds)`` steps.  At each step the
    sample is drawn from synthetic with probability ``synthetic_weight`` and
    from STPLS3D with probability ``1 - synthetic_weight``.  Both sub-datasets
    are sampled with replacement so that neither dominates the other purely by
    dataset-size difference.

    The expected ratio of synthetic : STPLS3D samples in one epoch is
    ``synthetic_weight : (1 - synthetic_weight)`` regardless of the raw sizes
    of the two datasets.

    Parameters
    ----------
    syn_ds : SceneChunkDataset
    stpls3d_ds : STPLS3DChunkDataset
    synthetic_weight : float
        Fraction of samples drawn from synthetic (e.g. 0.3 means 30% synthetic,
        70% STPLS3D).
    seed : int
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        syn_ds: SceneChunkDataset,
        stpls3d_ds: STPLS3DChunkDataset,
        synthetic_weight: float = 0.3,
        seed: int = 26011,
    ):
        self.syn_ds = syn_ds
        self.stpls3d_ds = stpls3d_ds
        self.synthetic_weight = float(synthetic_weight)
        self._rng = np.random.default_rng(seed)

        # Pre-generate an index sequence for one epoch.  Length = sum of both.
        # Re-generated at each epoch if the DataLoader uses a sampler.
        n_syn = int(len(syn_ds) * self.synthetic_weight / max(1.0 - self.synthetic_weight, 1e-9))
        n_syn = max(1, min(n_syn, len(syn_ds)))
        n_st = len(stpls3d_ds)
        self._epoch_len = n_syn + n_st

        # Build initial index mapping: (source, idx) where source 0=syn, 1=stpls3d
        self._build_index()

    def _build_index(self):
        n_syn = int(self._epoch_len * self.synthetic_weight)
        n_st = self._epoch_len - n_syn
        syn_idx = self._rng.integers(0, len(self.syn_ds), size=n_syn)
        st_idx = self._rng.integers(0, len(self.stpls3d_ds), size=n_st)
        # Interleave deterministically
        self._index = list(zip([0] * n_syn, syn_idx.tolist())) + list(zip([1] * n_st, st_idx.tolist()))
        self._rng.shuffle(self._index)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int) -> dict:
        source, idx = self._index[i]
        if source == 0:
            return self.syn_ds[idx]
        return self.stpls3d_ds[idx]
