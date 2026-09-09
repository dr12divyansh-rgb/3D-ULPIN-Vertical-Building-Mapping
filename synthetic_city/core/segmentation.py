"""Interfaces for future learned segmentation (placeholder only).

This step does NOT train any model. These interfaces define where a learned
segmenter (PointNet++ / RandLA-Net / ...) will plug in later, so that the
reconstruction system can accept segmented building points independently of
how they were produced.
"""

from __future__ import annotations


class PointCloudSegmenter:
    """Semantic segmentation of a building point cloud.

    Future implementations: PointNet++, RandLA-Net, etc. The reconstruction
    pipeline should depend only on this interface, never on a concrete model.
    """

    def predict(self, point_cloud):
        """Segment a point cloud.

        Parameters
        ----------
        point_cloud : ndarray of shape (N, C)
            Point coordinates (and optionally additional per-point features).

        Returns
        -------
        labels : ndarray of shape (N,)
            Per-point semantic class ids.
        """
        raise NotImplementedError("PointCloudSegmenter.predict is a placeholder interface")
