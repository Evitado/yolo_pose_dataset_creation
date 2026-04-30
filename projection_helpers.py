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
import cv2

from config_dataset import (
    FLIP_VERTICAL_FOR_DRAW,
    KEYPOINT_AZ_SHIFT_COLS,
    IMAGE_RENDER_MODE,
    IMAGE_CHANNEL_FIELDS,
    MULTI_FIELD_CHANNEL_BALANCE,
    MULTI_FIELD_CHANNEL_BALANCE_MIN_GAIN,
    MULTI_FIELD_CHANNEL_BALANCE_MAX_GAIN,
    IMAGE_COLORMAP,
    BLUE_CHANNEL_MODE,
    BLUE_CHANNEL_GAMMA,
    FAR_BRIGHT_BOOST_ENABLE,
    FAR_BRIGHT_BOOST_FIELDS,
    FAR_BRIGHT_BOOST_STRENGTH,
    INTENSITY_BOOST_ENABLE,
    INTENSITY_BOOST_GAIN,
    INTENSITY_ROW_CORRECTION_ENABLE,
    REFLECTIVITY_ROW_CORRECTION_ENABLE,
    INTENSITY_ROW_CORRECTION_STRENGTH,
    INTENSITY_ROW_CORRECTION_SIGMA_ROWS,
    INTENSITY_ROW_CORRECTION_MAX_SHIFT,
    SINGLE_CHANNEL_FIELD,
    RANGE_BRIGHT_MIN_M,
    RANGE_BRIGHT_MAX_M,
    RANGE_BRIGHT_SOFT_EDGE_M,
    RANGE_BRIGHT_OUTSIDE_LEVEL,
    RANGE_BRIGHT_PEAK_LEVEL,
    RETINEX_MODE,
    RETINEX_SIGMAS,
    RETINEX_GAIN,
    RETINEX_OFFSET,
    RETINEX_EPS,
    RETINEX_CLAHE_ENABLE,
    RETINEX_CLAHE_ONLY_SSR,
    RETINEX_CLAHE_CLIP_LIMIT,
    RETINEX_CLAHE_TILE_GRID,
)

FAR_BRIGHT_BOOST_FIELD_SET = {
    str(name).strip().lower()
    for name in FAR_BRIGHT_BOOST_FIELDS
    if str(name).strip()
}


def _norm_name(s: str) -> str:
    return s.lower().replace("-", "_").replace(" ", "_")


def find_alias(names: List[str], aliases: List[str]) -> Optional[str]:
    """Return the first name in `names` matching any of the `aliases` (normalized), or None."""
    for a in aliases:
        for n in names:
            if _norm_name(a) == _norm_name(n):
                return n
    return None


def _dashcam_gray_v01(field_values: np.ndarray) -> np.ndarray:
    """
    Replicates dashcam viewer ImageViewer.drawImage processing:
      1) uint16 -> float via /65535
      2) sigma rescale using mean ± 2σ
      3) tone mapping x / (1 + x)

    Returns float values in [0, 1].
    """
    float_image = field_values.astype(np.float32) / 65535.0

    mean = float(np.mean(float_image))
    variance = float(np.mean((float_image - mean) ** 2))
    sigma = float(np.sqrt(variance))

    low = max(0.0, mean - 2.0 * sigma)
    high = mean + 2.0 * sigma

    with np.errstate(divide="ignore", invalid="ignore"):
        scaled = (float_image - low) / (high - low)
        toned = scaled / (1.0 + scaled)

    toned = np.nan_to_num(toned, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(toned, 0.0, 1.0)

def _normalize_percentile01(img: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.0) -> np.ndarray:
    x = np.nan_to_num(img.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(x, (p_lo, p_hi))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.clip(x, 0.0, 1.0)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def _retinex_single_scale(gray01: np.ndarray, sigma: float, eps: float) -> np.ndarray:
    x = np.clip(gray01.astype(np.float32), 0.0, 1.0)
    if sigma <= 0:
        blur = x
    else:
        blur = cv2.GaussianBlur(
            x,
            (0, 0),
            sigmaX=float(sigma),
            sigmaY=float(sigma),
            borderType=cv2.BORDER_REPLICATE,
        )
    return np.log(x + eps) - np.log(blur + eps)


def _apply_retinex(gray01: np.ndarray) -> np.ndarray:
    mode = str(RETINEX_MODE).strip().lower()
    if mode in ("", "off", "none", "false", "0"):
        return np.clip(gray01, 0.0, 1.0)

    sigmas = [float(s) for s in RETINEX_SIGMAS if float(s) > 0.0]
    if not sigmas:
        sigmas = [80.0]

    eps = float(RETINEX_EPS)
    if eps <= 0.0:
        eps = 1e-6

    if mode == "ssr":
        response = _retinex_single_scale(gray01, sigmas[0], eps)
    else:
        # default to MSR behavior for unknown/"msr" mode
        stack = [_retinex_single_scale(gray01, s, eps) for s in sigmas]
        response = np.mean(stack, axis=0).astype(np.float32)

    response = float(RETINEX_GAIN) * response + float(RETINEX_OFFSET)
    return _normalize_percentile01(response, 1.0, 99.0)


def _apply_clahe_gray01(gray01: np.ndarray, clip_limit: float, tile_grid: int) -> np.ndarray:
    x = np.clip(gray01.astype(np.float32), 0.0, 1.0)
    u8 = np.round(x * 255.0).astype(np.uint8)
    g = max(1, int(tile_grid))
    cl = max(0.01, float(clip_limit))
    clahe = cv2.createCLAHE(clipLimit=cl, tileGridSize=(g, g))
    out = clahe.apply(u8).astype(np.float32) / 255.0
    return np.clip(out, 0.0, 1.0)


def _apply_post_retinex_enhancement(gray01: np.ndarray) -> np.ndarray:
    if not bool(RETINEX_CLAHE_ENABLE):
        return gray01

    mode = str(RETINEX_MODE).strip().lower()
    if bool(RETINEX_CLAHE_ONLY_SSR) and mode != "ssr":
        return gray01

    return _apply_clahe_gray01(
        gray01,
        clip_limit=float(RETINEX_CLAHE_CLIP_LIMIT),
        tile_grid=int(RETINEX_CLAHE_TILE_GRID),
    )


def _colormap_from_stops(gray01: np.ndarray, stops: np.ndarray) -> np.ndarray:
    x = np.clip(gray01, 0.0, 1.0)
    out = np.empty((*x.shape, 3), dtype=np.uint8)
    out[..., 0] = np.round(np.interp(x, stops[:, 0], stops[:, 1])).astype(np.uint8)
    out[..., 1] = np.round(np.interp(x, stops[:, 0], stops[:, 2])).astype(np.uint8)
    out[..., 2] = np.round(np.interp(x, stops[:, 0], stops[:, 3])).astype(np.uint8)
    return out


def _apply_colormap(gray01: np.ndarray, colormap: str) -> np.ndarray:
    mode = str(colormap).strip().lower()
    x = np.clip(gray01, 0.0, 1.0)

    if mode == "viridis":
        stops = np.array([
            [0.0, 68, 1, 84],
            [0.25, 59, 82, 139],
            [0.5, 33, 145, 140],
            [0.75, 94, 201, 98],
            [1.0, 253, 231, 37],
        ], dtype=np.float32)
        return _colormap_from_stops(x, stops)

    if mode == "plasma":
        stops = np.array([
            [0.0, 13, 8, 135],
            [0.25, 126, 3, 167],
            [0.5, 203, 71, 119],
            [0.75, 248, 149, 64],
            [1.0, 240, 249, 33],
        ], dtype=np.float32)
        return _colormap_from_stops(x, stops)

    if mode == "jet":
        r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
        g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
        b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
        return np.stack([
            np.round(r * 255).astype(np.uint8),
            np.round(g * 255).astype(np.uint8),
            np.round(b * 255).astype(np.uint8),
        ], axis=-1)

    # grayscale fallback
    v = np.round(x * 255).astype(np.uint8)
    return np.stack([v, v, v], axis=-1)


def _range_to_gray01(values: np.ndarray, mode: str, gamma: float) -> np.ndarray:
    vals = values.astype(np.float32)
    valid = np.isfinite(vals) & (vals > 0.0)
    out = np.zeros(vals.shape, dtype=np.float32)
    if not np.any(valid):
        return out

    vals_valid = vals[valid]
    m = str(mode).strip().lower()
    if m in {"bright_band", "range_bright_band", "band_bright"}:
        lo_m = float(min(RANGE_BRIGHT_MIN_M, RANGE_BRIGHT_MAX_M))
        hi_m = float(max(RANGE_BRIGHT_MIN_M, RANGE_BRIGHT_MAX_M))
        soft = max(1e-6, float(RANGE_BRIGHT_SOFT_EDGE_M))
        outside = float(np.clip(RANGE_BRIGHT_OUTSIDE_LEVEL, 0.0, 1.0))
        peak = float(np.clip(RANGE_BRIGHT_PEAK_LEVEL, outside, 1.0))

        base_valid = np.full(vals_valid.shape, outside, dtype=np.float32)

        core = (vals_valid >= lo_m) & (vals_valid <= hi_m)
        base_valid[core] = peak

        left = (vals_valid >= (lo_m - soft)) & (vals_valid < lo_m)
        if np.any(left):
            t = (vals_valid[left] - (lo_m - soft)) / soft
            base_valid[left] = outside + t * (peak - outside)

        right = (vals_valid > hi_m) & (vals_valid <= (hi_m + soft))
        if np.any(right):
            t = (vals_valid[right] - hi_m) / soft
            base_valid[right] = peak - t * (peak - outside)
    elif m == "log_range":
        work = np.log(vals_valid)
        lo, hi = np.percentile(work, (1.0, 99.0))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            base_valid = np.clip(work, 0.0, 1.0)
        else:
            base_valid = np.clip((work - lo) / (hi - lo), 0.0, 1.0)
    elif m == "inverse_range":
        lo, hi = np.percentile(vals_valid, (1.0, 99.0))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            base_valid = np.clip(vals_valid, 0.0, 1.0)
        else:
            base_valid = np.clip((vals_valid - lo) / (hi - lo), 0.0, 1.0)
        base_valid = 1.0 - base_valid
    elif m == "range":
        lo, hi = np.percentile(vals_valid, (1.0, 99.0))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            base_valid = np.clip(vals_valid, 0.0, 1.0)
        else:
            base_valid = np.clip((vals_valid - lo) / (hi - lo), 0.0, 1.0)
    else:
        # default: near brighter, far darker
        work = np.log(vals_valid)
        lo, hi = np.percentile(work, (1.0, 99.0))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            base_valid = np.clip(work, 0.0, 1.0)
        else:
            base_valid = np.clip((work - lo) / (hi - lo), 0.0, 1.0)
        base_valid = 1.0 - base_valid

    g = max(float(gamma), 1e-6)
    if abs(g - 1.0) > 1e-6:
        base_valid = np.power(np.clip(base_valid, 0.0, 1.0), g)

    out[valid] = np.clip(base_valid, 0.0, 1.0)
    return out


def _compute_far_boost_from_range(flat: np.ndarray, idx: dict[str, int]) -> Optional[np.ndarray]:
    if not bool(FAR_BRIGHT_BOOST_ENABLE):
        return None
    if "range" not in idx:
        return None
    boost = _range_to_gray01(flat[:, idx["range"]], BLUE_CHANNEL_MODE, BLUE_CHANNEL_GAMMA)
    return np.clip(boost, 0.0, 1.0)


def _apply_intensity_row_correction(gray01_2d: np.ndarray) -> np.ndarray:
    """
    Light row-wise correction for ring-like horizontal bands.
    """
    if not bool(INTENSITY_ROW_CORRECTION_ENABLE):
        return gray01_2d

    x = np.clip(np.asarray(gray01_2d, dtype=np.float32), 0.0, 1.0)
    if x.ndim != 2 or x.shape[0] < 3:
        return x

    strength = float(np.clip(INTENSITY_ROW_CORRECTION_STRENGTH, 0.0, 1.5))
    sigma = max(0.0, float(INTENSITY_ROW_CORRECTION_SIGMA_ROWS))
    if strength <= 0.0 or sigma <= 0.0:
        return x

    row_profile = np.nanmedian(x, axis=1).astype(np.float32)
    if not np.all(np.isfinite(row_profile)):
        m = np.isfinite(row_profile)
        if not np.any(m):
            return x
        fill = float(np.nanmedian(row_profile[m]))
        row_profile = np.where(m, row_profile, fill)

    trend = cv2.GaussianBlur(
        row_profile.reshape(-1, 1),
        (0, 0),
        sigmaX=0.0,
        sigmaY=sigma,
        borderType=cv2.BORDER_REPLICATE,
    ).reshape(-1)

    band = row_profile - trend
    max_shift = max(0.0, float(INTENSITY_ROW_CORRECTION_MAX_SHIFT))
    if max_shift > 0.0:
        band = np.clip(band, -max_shift, max_shift)

    corrected = x - strength * band[:, None]
    corrected += float(np.mean(x) - np.mean(corrected))
    return np.clip(corrected, 0.0, 1.0)


def _needs_row_correction(field_name: str) -> bool:
    f = str(field_name).strip().lower()
    if f == "intensity":
        return bool(INTENSITY_ROW_CORRECTION_ENABLE)
    if f == "reflectivity":
        return bool(REFLECTIVITY_ROW_CORRECTION_ENABLE)
    return False


def _field_to_gray01(
    flat: np.ndarray,
    idx: dict[str, int],
    field_name: str,
    far_boost01: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    f = str(field_name).strip().lower()
    if f not in idx:
        return None
    vals = flat[:, idx[f]]
    if f == "range":
        mode = str(BLUE_CHANNEL_MODE).strip().lower()
        if mode == "dashcam":
            return _dashcam_gray_v01(vals)
        return _range_to_gray01(vals, BLUE_CHANNEL_MODE, BLUE_CHANNEL_GAMMA)
    if f in {"reflectivity", "intensity", "signal", "ambient"}:
        base = _dashcam_gray_v01(vals)
        if (
            far_boost01 is not None
            and f in FAR_BRIGHT_BOOST_FIELD_SET
            and bool(FAR_BRIGHT_BOOST_ENABLE)
            and FAR_BRIGHT_BOOST_STRENGTH > 0
        ):
            a = float(np.clip(FAR_BRIGHT_BOOST_STRENGTH, 0.0, 1.5))
            base = np.clip(base + a * far_boost01 * (1.0 - base), 0.0, 1.0)
        if f == "intensity" and bool(INTENSITY_BOOST_ENABLE):
            g = max(0.0, float(INTENSITY_BOOST_GAIN))
            if g != 1.0:
                base = np.clip(base * g, 0.0, 1.0)
        return base
    return _normalize_percentile01(vals)


def _build_single_field_colormap(flat: np.ndarray, idx: dict[str, int], H: int, W: int) -> Optional[np.ndarray]:
    source_field = None
    for candidate in ("range", "signal", "intensity", "reflectivity"):
        if candidate in idx:
            source_field = candidate
            break
    if source_field is None:
        return None

    field = flat[:, idx[source_field]]
    gray01 = _dashcam_gray_v01(field).reshape(H, W)
    if _needs_row_correction(source_field):
        gray01 = _apply_intensity_row_correction(gray01)
    gray01 = _apply_retinex(gray01)
    gray01 = _apply_post_retinex_enhancement(gray01)
    return _apply_colormap(gray01, IMAGE_COLORMAP)


def _build_multi_field_rgb(flat: np.ndarray, idx: dict[str, int], H: int, W: int) -> Optional[np.ndarray]:
    fields = tuple(str(f).strip().lower() for f in IMAGE_CHANNEL_FIELDS)
    if len(fields) != 3:
        return None

    far_boost01 = _compute_far_boost_from_range(flat, idx)
    chans01 = []
    for f in fields:
        gray01 = _field_to_gray01(flat, idx, f, far_boost01=far_boost01)
        if gray01 is None:
            return None
        ch2d = np.clip(gray01, 0.0, 1.0).reshape(H, W)
        if _needs_row_correction(f):
            ch2d = _apply_intensity_row_correction(ch2d)
        chans01.append(ch2d)

    if bool(MULTI_FIELD_CHANNEL_BALANCE):
        # Rebalance channels per scene so one field cannot dominate globally.
        means = np.array([float(np.mean(ch)) for ch in chans01], dtype=np.float32)
        valid = means > 1e-6
        if np.count_nonzero(valid) >= 2:
            target = float(np.median(means[valid]))
            if target > 0.0:
                min_gain = max(1e-6, float(MULTI_FIELD_CHANNEL_BALANCE_MIN_GAIN))
                max_gain = max(min_gain, float(MULTI_FIELD_CHANNEL_BALANCE_MAX_GAIN))
                gains = np.ones(3, dtype=np.float32)
                gains[valid] = target / means[valid]
                gains = np.clip(gains, min_gain, max_gain)
                chans01 = [np.clip(ch * float(g), 0.0, 1.0) for ch, g in zip(chans01, gains)]

    chans_u8 = [np.round(ch * 255.0).astype(np.uint8) for ch in chans01]
    return np.stack(chans_u8, axis=-1)


def build_rgb_from_cols(flat: np.ndarray, cols: List[str], H: int, W: int) -> Optional[np.ndarray]:
    """
    Build a dataset image from lidar columns.

    Modes:
      - single_field_colormap: one lidar field with colormap and optional Retinex.
      - multi_field_rgb: map configured fields directly into RGB channels.

    If multi-field mode is requested but required fields are missing, this falls
    back to single-field colormap mode.
    """
    idx = {str(c).strip().lower(): i for i, c in enumerate(cols)}
    mode = str(IMAGE_RENDER_MODE).strip().lower()

    if mode in {"multi_field_rgb", "multi_field", "multichannel_rgb"}:
        rgb = _build_multi_field_rgb(flat, idx, H, W)
        if rgb is not None:
            return rgb

    return _build_single_field_colormap(flat, idx, H, W)


def build_gray_from_cols(flat: np.ndarray, cols: List[str], H: int, W: int) -> Optional[np.ndarray]:
    """
    Build a single-channel uint8 image (H, W) from a lidar field.
    """
    idx = {str(c).strip().lower(): i for i, c in enumerate(cols)}

    preferred = str(SINGLE_CHANNEL_FIELD).strip().lower()
    candidates: list[str] = []
    if preferred:
        candidates.append(preferred)
    for c in ("range", "signal", "intensity", "reflectivity", "ambient"):
        if c not in candidates:
            candidates.append(c)

    far_boost01 = _compute_far_boost_from_range(flat, idx)
    gray01 = None
    selected_field = ""
    for c in candidates:
        gray01 = _field_to_gray01(flat, idx, c, far_boost01=far_boost01)
        if gray01 is not None:
            selected_field = c
            break

    if gray01 is None:
        return None

    gray2d = np.clip(gray01, 0.0, 1.0).reshape(H, W)
    if _needs_row_correction(selected_field):
        gray2d = _apply_intensity_row_correction(gray2d)
    return np.round(gray2d * 255.0).astype(np.uint8)



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
