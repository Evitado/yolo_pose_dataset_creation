#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Projection / math helpers for YOLO-pose aircraft dataset creation.

- Normalization and RGB construction
- Elevation / azimuth computation
- Row/column mapping
- Bounding box helpers
- Ray-tracing depth utility
- Midpoint cluster adjustment for 'front_wheels_mid'
"""

from typing import List, Tuple, Optional

import numpy as np

from config_dataset import FLIP_VERTICAL_FOR_DRAW, KEYPOINT_AZ_SHIFT_COLS


def _norm_name(s: str) -> str:
    return s.lower().replace("-", "_").replace(" ", "_")


def find_alias(names: List[str], aliases: List[str]) -> Optional[str]:
    """Return the first name in `names` matching any of the `aliases` (normalized), or None."""
    for a in aliases:
        for n in names:
            if _norm_name(a) == _norm_name(n):
                return n
    return None


def _autoscale(img: np.ndarray, nan_fill: float = 0.0) -> np.ndarray:
    clean = np.nan_to_num(img, nan=nan_fill)
    vmin, vmax = np.percentile(clean, (1, 99))
    denom = (vmax - vmin) if (vmax > vmin) else 1.0
    return np.clip((clean - vmin) / (denom + 1e-12), 0.0, 1.0)


def _norm_uint8(x: np.ndarray) -> np.ndarray:
    return (_autoscale(x) * 255).astype(np.uint8)


def build_rgb_from_cols(flat: np.ndarray, cols: List[str], H: int, W: int) -> Optional[np.ndarray]:
    """
    Build an RGB image from point columns.

    Requires columns: 'reflectivity', 'range', 'intensity'.
    """
    idx = {c: i for i, c in enumerate(cols)}
    if all(k in idx for k in ("reflectivity", "range", "intensity")):
        r = _norm_uint8(flat[:, idx["reflectivity"]]).reshape(H, W)
        g = _norm_uint8(flat[:, idx["range"]]).reshape(H, W)
        b = _norm_uint8(flat[:, idx["intensity"]]).reshape(H, W)
        return np.stack([r, g, b], axis=-1)
    return None


def angles_from_xyz(xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute azimuth and elevation for each 3D point."""
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    az = np.arctan2(y, x)
    el = np.arctan2(z, np.sqrt(x * x + y * y))
    return az, el


def _circ_dist_array(a: float, arr: np.ndarray) -> np.ndarray:
    return np.abs(np.arctan2(np.sin(a - arr), np.cos(a - arr)))


def apply_transform(kps_xyz: np.ndarray, T: Optional[np.ndarray]) -> np.ndarray:
    """Apply homogeneous transform T (4×4) to Nx3 keypoints."""
    if T is None:
        return kps_xyz
    ones = np.ones((kps_xyz.shape[0], 1), dtype=np.float64)
    h = np.concatenate([kps_xyz, ones], axis=1)
    return (T @ h.T).T[:, :3]


def bbox_from_mask(mask: np.ndarray):
    """Return (x1, y1, x2, y2) bbox from a boolean mask, or None if empty."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def xyxy_to_xywhn(x1, y1, x2, y2, W, H):
    """Convert pixel bbox (x1,y1,x2,y2) to normalized YOLO (cx,cy,w,h)."""
    bw = max((x2 - x1 + 1), 1)
    bh = max((y2 - y1 + 1), 1)
    cx = (x1 + x2 + 1) / 2.0 / W
    cy = (y1 + y2 + 1) / 2.0 / H
    return float(cx), float(cy), float(bw / W), float(bh / H)


def _clip01(v: float) -> float:
    return float(np.clip(v, 0.0, 1.0))


def rc_to_xy_norm(r: int, c: int, H: int, W: int) -> Tuple[float, float]:
    """Convert row/col to normalized (x,y) in [0,1]."""
    return _clip01(c / W), _clip01(r / H)


def _fill_nans(arr: np.ndarray) -> np.ndarray:
    x = arr.copy()
    m = ~np.isfinite(x)
    if np.any(m):
        idx = np.where(~m)[0]
        if idx.size == 0:
            x[:] = 0.0
        else:
            for i in range(len(x)):
                j = idx[np.argmin(np.abs(idx - i))]
                x[i] = x[j]
    return x


def row_from_elevation(el_value: float, el_per_row: np.ndarray, H: int) -> int:
    """Map elevation value to image row index."""
    r = int(np.argmin(np.abs(el_per_row - el_value)))
    if FLIP_VERTICAL_FOR_DRAW:
        r = (H - 1) - r
    return int(np.clip(r, 0, H - 1))


def col_from_azimuth_global(az_value: float, az_per_col: np.ndarray, W: int) -> int:
    """Map azimuth to image column index using circular distance."""
    d = _circ_dist_array(az_value, az_per_col)
    c = int(np.nanargmin(d))
    return int((c + KEYPOINT_AZ_SHIFT_COLS) % W)


def get_min_depth(
    range_img: np.ndarray,
    valid_range: np.ndarray,
    r: int,
    c: int,
    patch_radius: int,
) -> Optional[float]:
    """
    Return min valid depth in a (2*patch_radius+1)² patch around (r,c).
    If nothing valid → None.
    """
    H, W = range_img.shape
    r0 = max(0, r - patch_radius)
    r1 = min(H, r + patch_radius + 1)
    c0 = max(0, c - patch_radius)
    c1 = min(W, c + patch_radius + 1)

    patch_valid = valid_range[r0:r1, c0:c1]
    if not np.any(patch_valid):
        return None

    patch_depths = range_img[r0:r1, c0:c1][patch_valid]
    return float(np.nanmin(patch_depths))


def adjust_midpoint_to_cluster(
    mid_xyz: np.ndarray,
    aircraft_pts: np.ndarray,
    z_min: float,
    base_radius: float,
    expand_radius: float,
    z_band: float,
    min_points: int,
) -> Tuple[np.ndarray, bool]:
    """
    Given raw midpoint mid_xyz and aircraft_pts (N,3), try to snap midpoint
    to a local 3D cluster near ground:

    1) small sphere of radius base_radius and z < z_min + z_band
    2) if not enough points, expand radius to expand_radius

    Returns (adjusted_midpoint, used_cluster_flag).
    """
    if aircraft_pts.size == 0:
        return mid_xyz, False

    diffs = aircraft_pts - mid_xyz.reshape(1, 3)
    dists = np.linalg.norm(diffs, axis=1)
    z_cond = aircraft_pts[:, 2] < (z_min + z_band)

    # Step 1: small radius
    mask_small = (dists < base_radius) & z_cond
    if np.count_nonzero(mask_small) >= min_points:
        cluster = aircraft_pts[mask_small]
        return cluster.mean(axis=0), True

    # Step 2: expanded radius
    mask_large = (dists < expand_radius) & z_cond
    if np.count_nonzero(mask_large) >= min_points:
        cluster = aircraft_pts[mask_large]
        return cluster.mean(axis=0), True

    # No decent cluster found; keep raw midpoint
    return mid_xyz, False
