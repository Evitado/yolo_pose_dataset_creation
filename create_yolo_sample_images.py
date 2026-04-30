#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create sample YOLO input images for every RGB channel combination of
reflectivity/range/intensity (with repetition), plus a default reference image
using the same renderer as create_yolo_pose_dataset.py.

Outputs are written under:
  <out_dir>/<combo_name>/<h5_stem>__<scene_name>.png
  <out_dir>/hsv/<combo_name>/<h5_stem>__<scene_name>.png
  <out_dir>/hsl/<combo_name>/<h5_stem>__<scene_name>.png
  <out_dir>/lab/<combo_name>/<h5_stem>__<scene_name>.png
  <out_dir>/canny/<combo_name>/<h5_stem>__<scene_name>.png
  <out_dir>/bilateral/<combo_name>/<h5_stem>__<scene_name>.png
  <out_dir>/clahe/<combo_name>/<h5_stem>__<scene_name>.png
  <out_dir>/bilateral_clahe/<combo_name>/<h5_stem>__<scene_name>.png
  <out_dir>/gray_clahe_sharpen/<combo_name>/<h5_stem>__<scene_name>.png
  <out_dir>/gray_clahe_sharpen_canny/<combo_name>/<h5_stem>__<scene_name>.png
  <out_dir>/gray_clahe_canny_morph/<combo_name>/<h5_stem>__<scene_name>.png
  <out_dir>/lab_l_clahe_bilateral/<combo_name>/<h5_stem>__<scene_name>.png
  <out_dir>/lab_l_clahe_bilateral_canny/<combo_name>/<h5_stem>__<scene_name>.png

  <out_dir>/default_rgb_like_dataset/<h5_stem>__<scene_name>.png
  <out_dir>/default_hsv_like_dataset/<h5_stem>__<scene_name>.png
  <out_dir>/default_hsl_like_dataset/<h5_stem>__<scene_name>.png
  <out_dir>/default_lab_like_dataset/<h5_stem>__<scene_name>.png
  <out_dir>/default_canny_like_dataset/<h5_stem>__<scene_name>.png
  <out_dir>/default_bilateral_like_dataset/<h5_stem>__<scene_name>.png
  <out_dir>/default_clahe_like_dataset/<h5_stem>__<scene_name>.png
  <out_dir>/default_bilateral_clahe_like_dataset/<h5_stem>__<scene_name>.png
  <out_dir>/default_gray_clahe_sharpen_like_dataset/<h5_stem>__<scene_name>.png
  <out_dir>/default_gray_clahe_sharpen_canny_like_dataset/<h5_stem>__<scene_name>.png
  <out_dir>/default_gray_clahe_canny_morph_like_dataset/<h5_stem>__<scene_name>.png
  <out_dir>/default_lab_l_clahe_bilateral_like_dataset/<h5_stem>__<scene_name>.png
  <out_dir>/default_lab_l_clahe_bilateral_canny_like_dataset/<h5_stem>__<scene_name>.png
"""

import argparse
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import h5py
import imageio.v2 as imageio
import numpy as np

from config_dataset import SOURCE, APPLY_MEDIAN_FILTER, MEDIAN_KSIZE
from io_helpers import list_h5_paths, open_h5_any
from projection_helpers import build_rgb_from_cols


FIELDS = ("reflectivity", "range", "intensity")

DEFAULT_REF_DIR = "default_rgb_like_dataset"
DEFAULT_REF_HSV_DIR = "default_hsv_like_dataset"
DEFAULT_REF_HSL_DIR = "default_hsl_like_dataset"
DEFAULT_REF_LAB_DIR = "default_lab_like_dataset"
DEFAULT_REF_CANNY_DIR = "default_canny_like_dataset"
DEFAULT_REF_BILATERAL_DIR = "default_bilateral_like_dataset"
DEFAULT_REF_CLAHE_DIR = "default_clahe_like_dataset"
DEFAULT_REF_BILATERAL_CLAHE_DIR = "default_bilateral_clahe_like_dataset"
DEFAULT_REF_GRAY_CLAHE_SHARPEN_DIR = "default_gray_clahe_sharpen_like_dataset"
DEFAULT_REF_GRAY_CLAHE_SHARPEN_CANNY_DIR = "default_gray_clahe_sharpen_canny_like_dataset"
DEFAULT_REF_GRAY_CLAHE_CANNY_MORPH_DIR = "default_gray_clahe_canny_morph_like_dataset"
DEFAULT_REF_LAB_L_CLAHE_BILATERAL_DIR = "default_lab_l_clahe_bilateral_like_dataset"
DEFAULT_REF_LAB_L_CLAHE_BILATERAL_CANNY_DIR = "default_lab_l_clahe_bilateral_canny_like_dataset"

HSV_ROOT_DIR = "hsv"
HSL_ROOT_DIR = "hsl"
LAB_ROOT_DIR = "lab"
CANNY_ROOT_DIR = "canny"
BILATERAL_ROOT_DIR = "bilateral"
CLAHE_ROOT_DIR = "clahe"
BILATERAL_CLAHE_ROOT_DIR = "bilateral_clahe"
GRAY_CLAHE_SHARPEN_ROOT_DIR = "gray_clahe_sharpen"
GRAY_CLAHE_SHARPEN_CANNY_ROOT_DIR = "gray_clahe_sharpen_canny"
GRAY_CLAHE_CANNY_MORPH_ROOT_DIR = "gray_clahe_canny_morph"
LAB_L_CLAHE_BILATERAL_ROOT_DIR = "lab_l_clahe_bilateral"
LAB_L_CLAHE_BILATERAL_CANNY_ROOT_DIR = "lab_l_clahe_bilateral_canny"


def _autoscale(arr: np.ndarray) -> np.ndarray:
    clean = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    vmin, vmax = np.percentile(clean, (1, 99))
    denom = (vmax - vmin) if (vmax > vmin) else 1.0
    return np.clip((clean - vmin) / (denom + 1e-12), 0.0, 1.0)


def _channel_from_field(flat: np.ndarray, idx: Dict[str, int], field: str, H: int, W: int) -> np.ndarray:
    vals = flat[:, idx[field]]
    if field == "range":
        vals = np.log(np.clip(vals, 1e-3, None))
    return (_autoscale(vals) * 255).astype(np.uint8).reshape(H, W)


def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)


def _rgb_to_hsl(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2HLS)


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)


def _rgb_to_canny(img: np.ndarray, canny_threshold1: float, canny_threshold2: float) -> np.ndarray:
    if img.ndim == 2:
        gray = img
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, threshold1=float(canny_threshold1), threshold2=float(canny_threshold2))
    return np.stack([edges, edges, edges], axis=-1)


def _apply_bilateral(
    rgb: np.ndarray,
    bilateral_d: int,
    bilateral_sigma_color: float,
    bilateral_sigma_space: float,
) -> np.ndarray:
    return cv2.bilateralFilter(
        rgb,
        d=int(bilateral_d),
        sigmaColor=float(bilateral_sigma_color),
        sigmaSpace=float(bilateral_sigma_space),
    )


def _apply_clahe_rgb(rgb: np.ndarray, clahe_clip_limit: float, clahe_tile_grid: int) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=float(clahe_clip_limit),
        tileGridSize=(int(clahe_tile_grid), int(clahe_tile_grid)),
    )
    l_eq = clahe.apply(l)
    out = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(out, cv2.COLOR_LAB2RGB)


def _gray_clahe_sharpen(
    rgb: np.ndarray,
    clahe_clip_limit: float,
    clahe_tile_grid: int,
    sharpen_amount: float,
    sharpen_sigma: float,
) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(
        clipLimit=float(clahe_clip_limit),
        tileGridSize=(int(clahe_tile_grid), int(clahe_tile_grid)),
    )
    gray_clahe = clahe.apply(gray)

    amt = max(0.0, float(sharpen_amount))
    sig = max(0.1, float(sharpen_sigma))
    if amt > 0.0:
        blur = cv2.GaussianBlur(gray_clahe, (0, 0), sigmaX=sig, sigmaY=sig)
        sharp = cv2.addWeighted(gray_clahe, 1.0 + amt, blur, -amt, 0)
    else:
        sharp = gray_clahe

    return np.stack([sharp, sharp, sharp], axis=-1)


def _gray_clahe_canny_morph(
    rgb: np.ndarray,
    clahe_clip_limit: float,
    clahe_tile_grid: int,
    canny_threshold1: float,
    canny_threshold2: float,
    dilate_kernel: int,
    dilate_iterations: int,
    close_kernel: int,
    close_iterations: int,
    open_enabled: bool,
    open_kernel: int,
    open_iterations: int,
) -> np.ndarray:
    """
    Pipeline:
      RGB -> grayscale -> CLAHE -> Canny -> dilation -> closing -> optional opening
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(
        clipLimit=float(clahe_clip_limit),
        tileGridSize=(int(clahe_tile_grid), int(clahe_tile_grid)),
    )
    gray_clahe = clahe.apply(gray)

    edges = cv2.Canny(
        gray_clahe,
        threshold1=float(canny_threshold1),
        threshold2=float(canny_threshold2),
    )

    d_k = max(1, int(dilate_kernel))
    d_it = max(1, int(dilate_iterations))
    kernel_d = np.ones((d_k, d_k), dtype=np.uint8)
    edges = cv2.dilate(edges, kernel_d, iterations=d_it)

    c_k = max(1, int(close_kernel))
    c_it = max(1, int(close_iterations))
    kernel_c = np.ones((c_k, c_k), dtype=np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_c, iterations=c_it)

    if bool(open_enabled):
        o_k = max(1, int(open_kernel))
        o_it = max(1, int(open_iterations))
        kernel_o = np.ones((o_k, o_k), dtype=np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_o, iterations=o_it)

    return np.stack([edges, edges, edges], axis=-1)


def _lab_l_clahe_bilateral(
    rgb: np.ndarray,
    clahe_clip_limit: float,
    clahe_tile_grid: int,
    bilateral_d: int,
    bilateral_sigma_color: float,
    bilateral_sigma_space: float,
) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l = lab[:, :, 0]

    clahe = cv2.createCLAHE(
        clipLimit=float(clahe_clip_limit),
        tileGridSize=(int(clahe_tile_grid), int(clahe_tile_grid)),
    )
    l_eq = clahe.apply(l)

    l_filt = cv2.bilateralFilter(
        l_eq,
        d=int(bilateral_d),
        sigmaColor=float(bilateral_sigma_color),
        sigmaSpace=float(bilateral_sigma_space),
    )

    return np.stack([l_filt, l_filt, l_filt], axis=-1)


def _build_combo_image(
    flat: np.ndarray,
    cols: List[str],
    H: int,
    W: int,
    combo: tuple[str, str, str],
) -> Optional[np.ndarray]:
    idx = {c: i for i, c in enumerate(cols)}
    if not all(f in idx for f in FIELDS):
        return None

    r = _channel_from_field(flat, idx, combo[0], H, W)
    g = _channel_from_field(flat, idx, combo[1], H, W)
    b = _channel_from_field(flat, idx, combo[2], H, W)
    img = np.stack([r, g, b], axis=-1)

    if APPLY_MEDIAN_FILTER:
        img = cv2.medianBlur(img, MEDIAN_KSIZE)
    return img


def _build_default_reference(flat: np.ndarray, cols: List[str], H: int, W: int) -> Optional[np.ndarray]:
    rgb = build_rgb_from_cols(flat, cols, H, W)
    if rgb is None:
        return None
    if APPLY_MEDIAN_FILTER:
        rgb = cv2.medianBlur(rgb.astype(np.uint8), MEDIAN_KSIZE)
    return rgb


def _compute_variants(
    img: np.ndarray,
    canny_threshold1: float,
    canny_threshold2: float,
    bilateral_d: int,
    bilateral_sigma_color: float,
    bilateral_sigma_space: float,
    clahe_clip_limit: float,
    clahe_tile_grid: int,
    sharpen_amount: float,
    sharpen_sigma: float,
    morph_dilate_kernel: int,
    morph_dilate_iterations: int,
    morph_close_kernel: int,
    morph_close_iterations: int,
    morph_open_enabled: bool,
    morph_open_kernel: int,
    morph_open_iterations: int,
) -> Dict[str, np.ndarray]:
    bilateral = _apply_bilateral(
        img,
        bilateral_d=bilateral_d,
        bilateral_sigma_color=bilateral_sigma_color,
        bilateral_sigma_space=bilateral_sigma_space,
    )
    clahe = _apply_clahe_rgb(img, clahe_clip_limit=clahe_clip_limit, clahe_tile_grid=clahe_tile_grid)
    bilateral_clahe = _apply_clahe_rgb(
        bilateral,
        clahe_clip_limit=clahe_clip_limit,
        clahe_tile_grid=clahe_tile_grid,
    )
    gray_clahe_sharpen = _gray_clahe_sharpen(
        img,
        clahe_clip_limit=clahe_clip_limit,
        clahe_tile_grid=clahe_tile_grid,
        sharpen_amount=sharpen_amount,
        sharpen_sigma=sharpen_sigma,
    )
    lab_l_clahe_bilateral = _lab_l_clahe_bilateral(
        img,
        clahe_clip_limit=clahe_clip_limit,
        clahe_tile_grid=clahe_tile_grid,
        bilateral_d=bilateral_d,
        bilateral_sigma_color=bilateral_sigma_color,
        bilateral_sigma_space=bilateral_sigma_space,
    )
    gray_clahe_sharpen_canny = _rgb_to_canny(
        gray_clahe_sharpen,
        canny_threshold1=canny_threshold1,
        canny_threshold2=canny_threshold2,
    )
    gray_clahe_canny_morph = _gray_clahe_canny_morph(
        img,
        clahe_clip_limit=clahe_clip_limit,
        clahe_tile_grid=clahe_tile_grid,
        canny_threshold1=canny_threshold1,
        canny_threshold2=canny_threshold2,
        dilate_kernel=morph_dilate_kernel,
        dilate_iterations=morph_dilate_iterations,
        close_kernel=morph_close_kernel,
        close_iterations=morph_close_iterations,
        open_enabled=morph_open_enabled,
        open_kernel=morph_open_kernel,
        open_iterations=morph_open_iterations,
    )
    lab_l_clahe_bilateral_canny = _rgb_to_canny(
        lab_l_clahe_bilateral,
        canny_threshold1=canny_threshold1,
        canny_threshold2=canny_threshold2,
    )

    return {
        "rgb": img,
        "hsv": _rgb_to_hsv(img),
        "hsl": _rgb_to_hsl(img),
        "lab": _rgb_to_lab(img),
        "canny": _rgb_to_canny(img, canny_threshold1, canny_threshold2),
        "bilateral": bilateral,
        "clahe": clahe,
        "bilateral_clahe": bilateral_clahe,
        "gray_clahe_sharpen": gray_clahe_sharpen,
        "gray_clahe_sharpen_canny": gray_clahe_sharpen_canny,
        "gray_clahe_canny_morph": gray_clahe_canny_morph,
        "lab_l_clahe_bilateral": lab_l_clahe_bilateral,
        "lab_l_clahe_bilateral_canny": lab_l_clahe_bilateral_canny,
    }


def _save(path: Path, img: np.ndarray) -> None:
    imageio.imwrite(str(path), img, compress_level=1)


def run(
    source: str,
    out_dir: str,
    max_h5_files: Optional[int],
    max_scenes_per_file: Optional[int],
    canny_threshold1: float,
    canny_threshold2: float,
    bilateral_d: int,
    bilateral_sigma_color: float,
    bilateral_sigma_space: float,
    clahe_clip_limit: float,
    clahe_tile_grid: int,
    sharpen_amount: float,
    sharpen_sigma: float,
    morph_dilate_kernel: int,
    morph_dilate_iterations: int,
    morph_close_kernel: int,
    morph_close_iterations: int,
    morph_open_enabled: bool,
    morph_open_kernel: int,
    morph_open_iterations: int,
) -> None:
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    combos = list(product(FIELDS, repeat=3))
    print(f"[combos] total={len(combos)}")

    h5_paths = list_h5_paths(source)
    if not h5_paths:
        raise RuntimeError(f"No .h5 files found under: {source}")

    if max_h5_files is not None and max_h5_files > 0:
        h5_paths = h5_paths[:max_h5_files]

    counts = {
        "combo_rgb": 0,
        "combo_hsv": 0,
        "combo_hsl": 0,
        "combo_lab": 0,
        "combo_canny": 0,
        "combo_bilateral": 0,
        "combo_clahe": 0,
        "combo_bilateral_clahe": 0,
        "combo_gray_clahe_sharpen": 0,
        "combo_gray_clahe_sharpen_canny": 0,
        "combo_gray_clahe_canny_morph": 0,
        "combo_lab_l_clahe_bilateral": 0,
        "combo_lab_l_clahe_bilateral_canny": 0,
        "ref_rgb": 0,
        "ref_hsv": 0,
        "ref_hsl": 0,
        "ref_lab": 0,
        "ref_canny": 0,
        "ref_bilateral": 0,
        "ref_clahe": 0,
        "ref_bilateral_clahe": 0,
        "ref_gray_clahe_sharpen": 0,
        "ref_gray_clahe_sharpen_canny": 0,
        "ref_gray_clahe_canny_morph": 0,
        "ref_lab_l_clahe_bilateral": 0,
        "ref_lab_l_clahe_bilateral_canny": 0,
    }

    for i, h5p in enumerate(h5_paths, 1):
        print(f"[{i}/{len(h5_paths)}] {Path(h5p).name}")
        with open_h5_any(h5p) as f:
            H = int(f.attrs["height"])
            W = int(f.attrs["width"])

            scene_names = [
                name for name, grp in f.items()
                if isinstance(grp, h5py.Group) and "points" in grp
            ]
            if max_scenes_per_file is not None and max_scenes_per_file > 0:
                scene_names = scene_names[:max_scenes_per_file]

            for scene_name in scene_names:
                ds = f[scene_name]["points"]
                flat = ds[()]

                cols_raw = ds.attrs.get("columns", None)
                cols = [
                    c.decode("utf-8") if isinstance(c, (bytes, bytearray)) else str(c)
                    for c in (cols_raw if cols_raw is not None else [])
                ]
                if not cols:
                    cols = [
                        "x", "y", "z", "range", "intensity", "reflectivity",
                        "ambient", "is_ground", "is_aircraft",
                    ]

                unique_scene = f"{Path(h5p).stem}__{scene_name}"

                ref_img = _build_default_reference(flat, cols, H, W)
                if ref_img is not None:
                    ref_variants = _compute_variants(
                        ref_img,
                        canny_threshold1=canny_threshold1,
                        canny_threshold2=canny_threshold2,
                        bilateral_d=bilateral_d,
                        bilateral_sigma_color=bilateral_sigma_color,
                        bilateral_sigma_space=bilateral_sigma_space,
                        clahe_clip_limit=clahe_clip_limit,
                        clahe_tile_grid=clahe_tile_grid,
                        sharpen_amount=sharpen_amount,
                        sharpen_sigma=sharpen_sigma,
                        morph_dilate_kernel=morph_dilate_kernel,
                        morph_dilate_iterations=morph_dilate_iterations,
                        morph_close_kernel=morph_close_kernel,
                        morph_close_iterations=morph_close_iterations,
                        morph_open_enabled=morph_open_enabled,
                        morph_open_kernel=morph_open_kernel,
                        morph_open_iterations=morph_open_iterations,
                    )

                    ref_dirs = {
                        "ref_rgb": out_root / DEFAULT_REF_DIR,
                        "ref_hsv": out_root / DEFAULT_REF_HSV_DIR,
                        "ref_hsl": out_root / DEFAULT_REF_HSL_DIR,
                        "ref_lab": out_root / DEFAULT_REF_LAB_DIR,
                        "ref_canny": out_root / DEFAULT_REF_CANNY_DIR,
                        "ref_bilateral": out_root / DEFAULT_REF_BILATERAL_DIR,
                        "ref_clahe": out_root / DEFAULT_REF_CLAHE_DIR,
                        "ref_bilateral_clahe": out_root / DEFAULT_REF_BILATERAL_CLAHE_DIR,
                        "ref_gray_clahe_sharpen": out_root / DEFAULT_REF_GRAY_CLAHE_SHARPEN_DIR,
                        "ref_gray_clahe_sharpen_canny": out_root / DEFAULT_REF_GRAY_CLAHE_SHARPEN_CANNY_DIR,
                        "ref_gray_clahe_canny_morph": out_root / DEFAULT_REF_GRAY_CLAHE_CANNY_MORPH_DIR,
                        "ref_lab_l_clahe_bilateral": out_root / DEFAULT_REF_LAB_L_CLAHE_BILATERAL_DIR,
                        "ref_lab_l_clahe_bilateral_canny": out_root / DEFAULT_REF_LAB_L_CLAHE_BILATERAL_CANNY_DIR,
                    }
                    for d in ref_dirs.values():
                        d.mkdir(parents=True, exist_ok=True)

                    _save(ref_dirs["ref_rgb"] / f"{unique_scene}.png", ref_variants["rgb"])
                    _save(ref_dirs["ref_hsv"] / f"{unique_scene}.png", ref_variants["hsv"])
                    _save(ref_dirs["ref_hsl"] / f"{unique_scene}.png", ref_variants["hsl"])
                    _save(ref_dirs["ref_lab"] / f"{unique_scene}.png", ref_variants["lab"])
                    _save(ref_dirs["ref_canny"] / f"{unique_scene}.png", ref_variants["canny"])
                    _save(ref_dirs["ref_bilateral"] / f"{unique_scene}.png", ref_variants["bilateral"])
                    _save(ref_dirs["ref_clahe"] / f"{unique_scene}.png", ref_variants["clahe"])
                    _save(ref_dirs["ref_bilateral_clahe"] / f"{unique_scene}.png", ref_variants["bilateral_clahe"])
                    _save(ref_dirs["ref_gray_clahe_sharpen"] / f"{unique_scene}.png", ref_variants["gray_clahe_sharpen"])
                    _save(
                        ref_dirs["ref_gray_clahe_sharpen_canny"] / f"{unique_scene}.png",
                        ref_variants["gray_clahe_sharpen_canny"],
                    )
                    _save(
                        ref_dirs["ref_gray_clahe_canny_morph"] / f"{unique_scene}.png",
                        ref_variants["gray_clahe_canny_morph"],
                    )
                    _save(ref_dirs["ref_lab_l_clahe_bilateral"] / f"{unique_scene}.png", ref_variants["lab_l_clahe_bilateral"])
                    _save(
                        ref_dirs["ref_lab_l_clahe_bilateral_canny"] / f"{unique_scene}.png",
                        ref_variants["lab_l_clahe_bilateral_canny"],
                    )

                    counts["ref_rgb"] += 1
                    counts["ref_hsv"] += 1
                    counts["ref_hsl"] += 1
                    counts["ref_lab"] += 1
                    counts["ref_canny"] += 1
                    counts["ref_bilateral"] += 1
                    counts["ref_clahe"] += 1
                    counts["ref_bilateral_clahe"] += 1
                    counts["ref_gray_clahe_sharpen"] += 1
                    counts["ref_gray_clahe_sharpen_canny"] += 1
                    counts["ref_gray_clahe_canny_morph"] += 1
                    counts["ref_lab_l_clahe_bilateral"] += 1
                    counts["ref_lab_l_clahe_bilateral_canny"] += 1
                else:
                    print(f"  [WARN] {unique_scene}: default reference image could not be built")

                for combo in combos:
                    combo_img = _build_combo_image(flat, cols, H, W, combo)
                    if combo_img is None:
                        print(f"  [SKIP] {unique_scene}: missing one of {FIELDS}")
                        break

                    combo_variants = _compute_variants(
                        combo_img,
                        canny_threshold1=canny_threshold1,
                        canny_threshold2=canny_threshold2,
                        bilateral_d=bilateral_d,
                        bilateral_sigma_color=bilateral_sigma_color,
                        bilateral_sigma_space=bilateral_sigma_space,
                        clahe_clip_limit=clahe_clip_limit,
                        clahe_tile_grid=clahe_tile_grid,
                        sharpen_amount=sharpen_amount,
                        sharpen_sigma=sharpen_sigma,
                        morph_dilate_kernel=morph_dilate_kernel,
                        morph_dilate_iterations=morph_dilate_iterations,
                        morph_close_kernel=morph_close_kernel,
                        morph_close_iterations=morph_close_iterations,
                        morph_open_enabled=morph_open_enabled,
                        morph_open_kernel=morph_open_kernel,
                        morph_open_iterations=morph_open_iterations,
                    )

                    combo_name = f"{combo[0]}_{combo[1]}_{combo[2]}"
                    combo_dirs = {
                        "combo_rgb": out_root / combo_name,
                        "combo_hsv": out_root / HSV_ROOT_DIR / combo_name,
                        "combo_hsl": out_root / HSL_ROOT_DIR / combo_name,
                        "combo_lab": out_root / LAB_ROOT_DIR / combo_name,
                        "combo_canny": out_root / CANNY_ROOT_DIR / combo_name,
                        "combo_bilateral": out_root / BILATERAL_ROOT_DIR / combo_name,
                        "combo_clahe": out_root / CLAHE_ROOT_DIR / combo_name,
                        "combo_bilateral_clahe": out_root / BILATERAL_CLAHE_ROOT_DIR / combo_name,
                        "combo_gray_clahe_sharpen": out_root / GRAY_CLAHE_SHARPEN_ROOT_DIR / combo_name,
                        "combo_gray_clahe_sharpen_canny": out_root / GRAY_CLAHE_SHARPEN_CANNY_ROOT_DIR / combo_name,
                        "combo_gray_clahe_canny_morph": out_root / GRAY_CLAHE_CANNY_MORPH_ROOT_DIR / combo_name,
                        "combo_lab_l_clahe_bilateral": out_root / LAB_L_CLAHE_BILATERAL_ROOT_DIR / combo_name,
                        "combo_lab_l_clahe_bilateral_canny": out_root / LAB_L_CLAHE_BILATERAL_CANNY_ROOT_DIR / combo_name,
                    }
                    for d in combo_dirs.values():
                        d.mkdir(parents=True, exist_ok=True)

                    _save(combo_dirs["combo_rgb"] / f"{unique_scene}.png", combo_variants["rgb"])
                    _save(combo_dirs["combo_hsv"] / f"{unique_scene}.png", combo_variants["hsv"])
                    _save(combo_dirs["combo_hsl"] / f"{unique_scene}.png", combo_variants["hsl"])
                    _save(combo_dirs["combo_lab"] / f"{unique_scene}.png", combo_variants["lab"])
                    _save(combo_dirs["combo_canny"] / f"{unique_scene}.png", combo_variants["canny"])
                    _save(combo_dirs["combo_bilateral"] / f"{unique_scene}.png", combo_variants["bilateral"])
                    _save(combo_dirs["combo_clahe"] / f"{unique_scene}.png", combo_variants["clahe"])
                    _save(combo_dirs["combo_bilateral_clahe"] / f"{unique_scene}.png", combo_variants["bilateral_clahe"])
                    _save(combo_dirs["combo_gray_clahe_sharpen"] / f"{unique_scene}.png", combo_variants["gray_clahe_sharpen"])
                    _save(
                        combo_dirs["combo_gray_clahe_sharpen_canny"] / f"{unique_scene}.png",
                        combo_variants["gray_clahe_sharpen_canny"],
                    )
                    _save(
                        combo_dirs["combo_gray_clahe_canny_morph"] / f"{unique_scene}.png",
                        combo_variants["gray_clahe_canny_morph"],
                    )
                    _save(combo_dirs["combo_lab_l_clahe_bilateral"] / f"{unique_scene}.png", combo_variants["lab_l_clahe_bilateral"])
                    _save(
                        combo_dirs["combo_lab_l_clahe_bilateral_canny"] / f"{unique_scene}.png",
                        combo_variants["lab_l_clahe_bilateral_canny"],
                    )

                    counts["combo_rgb"] += 1
                    counts["combo_hsv"] += 1
                    counts["combo_hsl"] += 1
                    counts["combo_lab"] += 1
                    counts["combo_canny"] += 1
                    counts["combo_bilateral"] += 1
                    counts["combo_clahe"] += 1
                    counts["combo_bilateral_clahe"] += 1
                    counts["combo_gray_clahe_sharpen"] += 1
                    counts["combo_gray_clahe_sharpen_canny"] += 1
                    counts["combo_gray_clahe_canny_morph"] += 1
                    counts["combo_lab_l_clahe_bilateral"] += 1
                    counts["combo_lab_l_clahe_bilateral_canny"] += 1

    print(f"[done] wrote {counts['combo_rgb']} combo RGB images")
    print(f"[done] wrote {counts['combo_hsv']} combo HSV images")
    print(f"[done] wrote {counts['combo_hsl']} combo HSL images")
    print(f"[done] wrote {counts['combo_lab']} combo LAB images")
    print(f"[done] wrote {counts['combo_canny']} combo Canny images")
    print(f"[done] wrote {counts['combo_bilateral']} combo Bilateral images")
    print(f"[done] wrote {counts['combo_clahe']} combo CLAHE images")
    print(f"[done] wrote {counts['combo_bilateral_clahe']} combo Bilateral+CLAHE images")
    print(f"[done] wrote {counts['combo_gray_clahe_sharpen']} combo Gray+CLAHE+Sharpen images")
    print(f"[done] wrote {counts['combo_gray_clahe_sharpen_canny']} combo Gray+CLAHE+Sharpen+Canny images")
    print(f"[done] wrote {counts['combo_gray_clahe_canny_morph']} combo Gray+CLAHE+Canny+Dilate+Close(+Open) images")
    print(f"[done] wrote {counts['combo_lab_l_clahe_bilateral']} combo LAB-L+CLAHE+Bilateral images")
    print(f"[done] wrote {counts['combo_lab_l_clahe_bilateral_canny']} combo LAB-L+CLAHE+Bilateral+Canny images")

    print(f"[done] wrote {counts['ref_rgb']} default reference RGB images")
    print(f"[done] wrote {counts['ref_hsv']} default reference HSV images")
    print(f"[done] wrote {counts['ref_hsl']} default reference HSL images")
    print(f"[done] wrote {counts['ref_lab']} default reference LAB images")
    print(f"[done] wrote {counts['ref_canny']} default reference Canny images")
    print(f"[done] wrote {counts['ref_bilateral']} default reference Bilateral images")
    print(f"[done] wrote {counts['ref_clahe']} default reference CLAHE images")
    print(f"[done] wrote {counts['ref_bilateral_clahe']} default reference Bilateral+CLAHE images")
    print(f"[done] wrote {counts['ref_gray_clahe_sharpen']} default reference Gray+CLAHE+Sharpen images")
    print(f"[done] wrote {counts['ref_gray_clahe_sharpen_canny']} default reference Gray+CLAHE+Sharpen+Canny images")
    print(f"[done] wrote {counts['ref_gray_clahe_canny_morph']} default reference Gray+CLAHE+Canny+Dilate+Close(+Open) images")
    print(f"[done] wrote {counts['ref_lab_l_clahe_bilateral']} default reference LAB-L+CLAHE+Bilateral images")
    print(f"[done] wrote {counts['ref_lab_l_clahe_bilateral_canny']} default reference LAB-L+CLAHE+Bilateral+Canny images")
    print(f"[out] {out_root.resolve()}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Export RGB combinations from reflectivity/range/intensity plus "
            "HSV/HSL/LAB/Canny/Bilateral/CLAHE and grayscale/L-channel variants."
        )
    )
    ap.add_argument("--source", type=str, default=SOURCE, help="GCS prefix or local folder with .h5 files")
    ap.add_argument("--out-dir", type=str, default="./yolo sample images", help="Output folder")
    ap.add_argument("--max-h5-files", type=int, default=1, help="Limit number of H5 files")
    ap.add_argument("--max-scenes-per-file", type=int, default=3, help="Limit scenes per H5")

    ap.add_argument("--canny-threshold1", type=float, default=100.0, help="Canny lower threshold")
    ap.add_argument("--canny-threshold2", type=float, default=200.0, help="Canny upper threshold")

    ap.add_argument("--bilateral-d", type=int, default=7, help="Bilateral filter diameter")
    ap.add_argument("--bilateral-sigma-color", type=float, default=75.0, help="Bilateral sigmaColor")
    ap.add_argument("--bilateral-sigma-space", type=float, default=75.0, help="Bilateral sigmaSpace")

    ap.add_argument("--clahe-clip-limit", type=float, default=2.0, help="CLAHE clip limit")
    ap.add_argument("--clahe-tile-grid", type=int, default=8, help="CLAHE tile grid size NxN")

    ap.add_argument(
        "--sharpen-amount",
        type=float,
        default=0.25,
        help="Unsharp amount for grayscale CLAHE output (0.0 disables)",
    )
    ap.add_argument(
        "--sharpen-sigma",
        type=float,
        default=1.0,
        help="Gaussian sigma for unsharp mask blur",
    )
    ap.add_argument(
        "--morph-dilate-kernel",
        type=int,
        default=3,
        help="Kernel size for post-Canny dilation",
    )
    ap.add_argument(
        "--morph-dilate-iterations",
        type=int,
        default=1,
        help="Dilation iterations after Canny",
    )
    ap.add_argument(
        "--morph-close-kernel",
        type=int,
        default=3,
        help="Kernel size for morphological closing",
    )
    ap.add_argument(
        "--morph-close-iterations",
        type=int,
        default=1,
        help="Closing iterations after dilation",
    )
    ap.add_argument(
        "--morph-open-enable",
        action="store_true",
        help="Enable optional morphological opening after closing",
    )
    ap.add_argument(
        "--morph-open-kernel",
        type=int,
        default=3,
        help="Kernel size for optional opening",
    )
    ap.add_argument(
        "--morph-open-iterations",
        type=int,
        default=1,
        help="Opening iterations when enabled",
    )

    return ap.parse_args()


def main() -> None:
    args = parse_args()
    run(
        source=args.source,
        out_dir=args.out_dir,
        max_h5_files=args.max_h5_files,
        max_scenes_per_file=args.max_scenes_per_file,
        canny_threshold1=args.canny_threshold1,
        canny_threshold2=args.canny_threshold2,
        bilateral_d=args.bilateral_d,
        bilateral_sigma_color=args.bilateral_sigma_color,
        bilateral_sigma_space=args.bilateral_sigma_space,
        clahe_clip_limit=args.clahe_clip_limit,
        clahe_tile_grid=args.clahe_tile_grid,
        sharpen_amount=args.sharpen_amount,
        sharpen_sigma=args.sharpen_sigma,
        morph_dilate_kernel=args.morph_dilate_kernel,
        morph_dilate_iterations=args.morph_dilate_iterations,
        morph_close_kernel=args.morph_close_kernel,
        morph_close_iterations=args.morph_close_iterations,
        morph_open_enabled=args.morph_open_enable,
        morph_open_kernel=args.morph_open_kernel,
        morph_open_iterations=args.morph_open_iterations,
    )


if __name__ == "__main__":
    main()
