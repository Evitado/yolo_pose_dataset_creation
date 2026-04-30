#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run a YOLO-Pose .pt model on images built from H5 range images and
save pointclouds cropped to the predicted bbox as .pcd (ASCII).

This is similar to test_yolo_pose_h5_to_pcd.py, but adds:
- weights can be a directory (auto-picks best.pt or last.pt)
- optional debug image output
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import h5py
import cv2

from config_dataset import (
    SOURCE,
    APPLY_MEDIAN_FILTER,
    MEDIAN_KSIZE,
    ROLL_WIDE_BBOX,
    ROLL_WIDE_BBOX_FRAC,
    ROLL_WIDE_BBOX_COLS,
)
from io_helpers import list_h5_paths, open_h5_any
from projection_helpers import build_rgb_from_cols

DEFAULT_YAML_KP_NAMES = (
    "/home/femi/yolo_pose_dataset_creation/"
    "aircraft_pose_with_normalising_applied_multifield_only_3/aircraft_pose.yaml"
)
DRAW_PATH_TEXT_ON_IMAGE = False
DEBUG_KPT_RADIUS_PX = 8
DEBUG_KPT_SPHERE_RADIUS_M = 0.25
ALLOW_GLOBAL_KP_XYZ_FALLBACK = True
APPEND_KEYPOINTS_TO_PCD = True
DRAW_BBOX_ON_DEBUG_IMAGE = True
DRAW_3D_OBB = False

# Code-level path overrides (set USE_CODE_PATH_OVERRIDES=True to use these instead of CLI args)
USE_CODE_PATH_OVERRIDES = True
CODE_SOURCE_PATH = SOURCE
CODE_WEIGHTS_PATH = "/home/femi/yolo_pose_dataset_creation/runs/pose/aircraft_pose_new_clear_img8/weights/best.pt"
CODE_OUT_DIR = "/home/femi/yolo_pose_dataset_creation/pcd_from_yolo"
CODE_IMAGE_PATH: Optional[str] = None
CODE_IMAGE_DIR: Optional[str] = None
CODE_YAML_KP_NAMES = DEFAULT_YAML_KP_NAMES
CODE_KP_CONF_CSV: Optional[str] = None


def _load_keypoint_names_from_yaml(yaml_path: Optional[str]) -> List[str]:
    if not yaml_path:
        return []
    p = Path(yaml_path).expanduser()
    if not p.exists() or not p.is_file():
        return []
    lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    names: List[str] = []
    in_block = False
    for ln in lines:
        s = ln.strip()
        if s == "keypoints:":
            in_block = True
            continue
        if in_block:
            if s.startswith("- "):
                names.append(s[2:].strip())
            elif s and not s.startswith("#"):
                break
    return names


def _parse_unique_scene_stem(stem: str) -> Tuple[str, str]:
    """
    Parse '<h5_stem>__<scene_name>' into (h5_stem, scene_name).
    Uses rsplit because h5_stem itself contains '__'.
    """
    s = str(stem).strip()
    if "__" not in s:
        raise ValueError(
            f"Image stem '{s}' does not contain '__'. Expected '<h5_stem>__<scene_name>'."
        )
    h5_stem, scene_name = s.rsplit("__", 1)
    if not h5_stem or not scene_name:
        raise ValueError(
            f"Could not parse image stem '{s}' into h5 stem and scene name."
        )
    return h5_stem, scene_name


def _resolve_image_path_with_split_fallback(image_path: str) -> Path:
    """
    Resolve image path; if missing and path looks like .../images/<split>/<file>,
    try sibling splits train/val/test with same filename.
    """
    p = Path(image_path).expanduser().resolve()
    if p.exists() and p.is_file():
        return p

    fname = p.name
    parts = p.parts
    if "images" in parts:
        i = parts.index("images")
        base = Path(*parts[: i + 1])  # .../images
        for sp in ("test", "val", "train"):
            cand = base / sp / fname
            if cand.exists() and cand.is_file():
                print(f"[image-mode] --image-path not found, using fallback: {cand}")
                return cand.resolve()

    raise RuntimeError(f"--image-path not found: {p}")


def _collect_images_from_dir(image_dir: Path) -> List[Path]:
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    if not image_dir.exists() or not image_dir.is_dir():
        raise RuntimeError(f"--image-dir not found or not a directory: {image_dir}")
    imgs = sorted([p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_exts])
    if not imgs:
        raise RuntimeError(f"No images found in --image-dir: {image_dir}")
    return imgs


def _bbox_from_mask(mask2d: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.nonzero(mask2d)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _find_best_azimuth_roll(mask2d: np.ndarray) -> int:
    """
    Match create_yolo_pose_dataset seam placement:
    place seam (col 0) in the largest empty-azimuth gap.
    """
    _, W = mask2d.shape
    if W <= 1:
        return 0

    col_has_aircraft = mask2d.any(axis=0)
    empty = ~col_has_aircraft
    if not np.any(empty):
        return 0

    best_len = 0
    best_start = 0
    for start in range(W):
        if not empty[start]:
            continue
        length = 0
        while length < W and empty[(start + length) % W]:
            length += 1
        if length > best_len:
            best_len = length
            best_start = start

    if best_len <= 0:
        return 0

    seam_col = (best_start + best_len // 2) % W
    shift = -int(seam_col)
    if shift % W == 0:
        return 0
    return shift


def _wrap_aware_bbox(mask2d: np.ndarray) -> Tuple[Optional[Tuple[int, int, int, int]], int]:
    """
    Match create_yolo_pose_dataset wrap-aware bbox pre-roll.
    Returns (bbox_in_shifted_coords_or_original, shift_cols).
    """
    bb = _bbox_from_mask(mask2d)
    if bb is None:
        return None, 0

    W = mask2d.shape[1]
    if W <= 1:
        return bb, 0

    x1, _, x2, _ = bb
    w0 = x2 - x1 + 1

    shift = _find_best_azimuth_roll(mask2d)
    if shift == 0:
        return bb, 0

    rolled_mask = np.roll(mask2d, shift=shift, axis=1)
    bb2 = _bbox_from_mask(rolled_mask)
    if bb2 is None:
        return bb, 0

    w2 = bb2[2] - bb2[0] + 1
    if w2 < w0:
        return bb2, int(shift)
    return bb, 0


def _to_signed_roll(shift: int, W: int) -> int:
    if W <= 1:
        return 0
    s = int(shift) % int(W)
    if s > (W // 2):
        s -= int(W)
    return int(s)


def _extract_is_aircraft_mask(flat: np.ndarray, cols: List[str], H: int, W: int) -> Optional[np.ndarray]:
    col_to_idx = {str(c).strip().lower(): i for i, c in enumerate(cols)}
    idx = None
    for k in ("is_aircraft", "aircraft_mask", "mask_aircraft", "aircraft"):
        if k in col_to_idx:
            idx = col_to_idx[k]
            break
    if idx is None:
        return None
    raw = flat[:, idx].reshape(H, W)
    if raw.dtype == np.bool_:
        return raw.copy()
    return np.asarray(raw > 0.5, dtype=bool)


def _compute_export_like_roll(mask2d: np.ndarray) -> int:
    """
    Rebuild the same azimuth roll logic used during dataset export so that
    bbox/keypoints from exported images map onto the matching xyz columns.
    """
    if mask2d is None:
        return 0
    H, W = mask2d.shape
    if H <= 0 or W <= 1 or not bool(ROLL_WIDE_BBOX):
        return 0

    bb = _bbox_from_mask(mask2d)
    if bb is None:
        return 0

    x1, _, x2, _ = bb
    bbox_frac = (x2 - x1 + 1) / float(W)
    shift_total = 0
    m = mask2d

    bb_wrap, shift_wrap = _wrap_aware_bbox(m)
    if bb_wrap is not None and shift_wrap != 0:
        m = np.roll(m, shift=int(shift_wrap), axis=1)
        shift_total += int(shift_wrap)
        x1w, _, x2w, _ = bb_wrap
        bbox_frac = (x2w - x1w + 1) / float(W)

    if bbox_frac > float(ROLL_WIDE_BBOX_FRAC):
        shift = _find_best_azimuth_roll(m)
        if shift == 0:
            shift = int(ROLL_WIDE_BBOX_COLS) % int(W)
        if shift != 0:
            shift_total += int(shift)

    return _to_signed_roll(shift_total, W)


def _bbox_aircraft_coverage(
    mask2d: Optional[np.ndarray],
    bb: Tuple[int, int, int, int],
) -> Optional[Dict[str, float]]:
    """
    Compute how much of the aircraft mask is covered by bbox.
    Returns recall-like metric:
      aircraft_recall = aircraft_pixels_inside_bbox / aircraft_pixels_total
    """
    if mask2d is None:
        return None
    if mask2d.ndim != 2:
        return None
    H, W = mask2d.shape
    if H <= 0 or W <= 0:
        return None

    x1, y1, x2, y2 = bb
    x1 = int(np.clip(x1, 0, W - 1))
    x2 = int(np.clip(x2, 0, W - 1))
    y1 = int(np.clip(y1, 0, H - 1))
    y2 = int(np.clip(y2, 0, H - 1))
    if x2 < x1 or y2 < y1:
        return None

    m = np.asarray(mask2d, dtype=bool)
    total_aircraft = int(np.count_nonzero(m))
    if total_aircraft <= 0:
        return {
            "aircraft_px_total": 0.0,
            "aircraft_px_inside": 0.0,
            "aircraft_recall": 0.0,
            "bbox_area_px": float((x2 - x1 + 1) * (y2 - y1 + 1)),
        }

    inside = int(np.count_nonzero(m[y1 : y2 + 1, x1 : x2 + 1]))
    bbox_area = int((x2 - x1 + 1) * (y2 - y1 + 1))
    recall = float(inside) / float(total_aircraft)
    return {
        "aircraft_px_total": float(total_aircraft),
        "aircraft_px_inside": float(inside),
        "aircraft_recall": float(recall),
        "bbox_area_px": float(bbox_area),
    }


def write_pcd_xyz(path: Path, points_xyz: np.ndarray) -> None:
    """Write Nx3 pointcloud to ASCII .pcd."""
    pts = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    header = "\n".join(
        [
            "# .PCD v0.7 - Point Cloud Data file format",
            "VERSION 0.7",
            "FIELDS x y z",
            "SIZE 4 4 4",
            "TYPE F F F",
            "COUNT 1 1 1",
            f"WIDTH {pts.shape[0]}",
            "HEIGHT 1",
            "VIEWPOINT 0 0 0 1 0 0 0",
            f"POINTS {pts.shape[0]}",
            "DATA ascii",
        ]
    )

    with path.open("w", encoding="utf-8") as f:
        f.write(header + "\n")
        for x, y, z in pts:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def _pick_best_bbox(result, target_class: int = 0) -> Optional[Tuple[int, int, int, int]]:
    """Return (x1,y1,x2,y2) for the highest-confidence bbox of target_class."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None

    cls = boxes.cls.detach().cpu().numpy().astype(int)
    conf = boxes.conf.detach().cpu().numpy()
    xyxy = boxes.xyxy.detach().cpu().numpy()

    best_i = None
    best_c = -1.0
    for i in range(len(xyxy)):
        if cls[i] != target_class:
            continue
        if conf[i] > best_c:
            best_c = float(conf[i])
            best_i = i

    if best_i is None:
        return None

    x1, y1, x2, y2 = xyxy[best_i].tolist()
    return int(x1), int(y1), int(x2), int(y2)


def _pick_best_det_idx(result, target_class: int = 0) -> Optional[int]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None
    cls = boxes.cls.detach().cpu().numpy().astype(int)
    conf = boxes.conf.detach().cpu().numpy()
    best_i = None
    best_c = -1.0
    for i in range(len(cls)):
        if cls[i] != target_class:
            continue
        if float(conf[i]) > best_c:
            best_c = float(conf[i])
            best_i = i
    return best_i


def _build_rgb_and_xyz(flat: np.ndarray, cols: List[str], H: int, W: int):
    """Build RGB image and XYZ grid (H,W,3)."""
    if not {"x", "y", "z"}.issubset(set(cols)):
        return None, None
    ix, iy, iz = cols.index("x"), cols.index("y"), cols.index("z")

    rgb = build_rgb_from_cols(flat, cols, H, W)
    if rgb is None:
        return None, None

    if APPLY_MEDIAN_FILTER:
        rgb = cv2.medianBlur(rgb.astype(np.uint8), MEDIAN_KSIZE)

    xyz = np.stack([flat[:, ix], flat[:, iy], flat[:, iz]], axis=1).astype(np.float32)
    return rgb, xyz.reshape(H, W, 3)


def _resolve_weights(weights: str) -> Path:
    wp = Path(weights)
    if wp.is_dir():
        best = wp / "best.pt"
        last = wp / "last.pt"
        if best.exists():
            return best
        if last.exists():
            return last
        raise RuntimeError(f"No best.pt or last.pt found in: {wp}")
    return wp


def _draw_bbox(rgb: np.ndarray, bb: Tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bb
    img = rgb.copy()
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return img


def _draw_bbox_and_keypoints(
    rgb: np.ndarray,
    bb: Tuple[int, int, int, int],
    kp_xy: Optional[np.ndarray],
    kp_conf_arr: Optional[np.ndarray],
    kp_conf_thr: float,
    kp_names: List[str],
    draw_bbox: bool = True,
    path_lines: Optional[List[str]] = None,
) -> np.ndarray:
    img = _draw_bbox(rgb, bb) if draw_bbox else rgb.copy()
    if kp_xy is None:
        kp_iter = []
    else:
        kp_iter = list(enumerate(kp_xy))

    for k, (x, y) in kp_iter:
        if kp_conf_arr is not None and k < len(kp_conf_arr) and float(kp_conf_arr[k]) < float(kp_conf_thr):
            continue
        xi = int(round(float(x)))
        yi = int(round(float(y)))
        cv2.circle(
            img,
            (xi, yi),
            int(DEBUG_KPT_RADIUS_PX),
            (255, 0, 0),
            -1,
            lineType=cv2.LINE_AA,
        )
        name = kp_names[k] if k < len(kp_names) and kp_names[k] else f"K{k}"
        cv2.putText(
            img,
            name,
            (xi + 5, yi - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    if path_lines:
        y0 = 16
        for ln in path_lines:
            cv2.putText(
                img,
                str(ln),
                (6, y0),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            y0 += 15
    return img


def _sample_xyz_nearest(
    xyz_hw3: np.ndarray,
    r0: int,
    c0: int,
    radius: int = 3,
    allow_global_fallback: bool = False,
    mask_aircraft: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    def _depth_inlier_mask(pts_xyz: np.ndarray) -> np.ndarray:
        n = int(pts_xyz.shape[0])
        if n <= 0:
            return np.zeros((0,), dtype=bool)
        if n < 4:
            return np.ones((n,), dtype=bool)

        depth = np.linalg.norm(pts_xyz.astype(np.float64), axis=1)
        finite = np.isfinite(depth)
        if not np.any(finite):
            return np.zeros((n,), dtype=bool)
        if not np.all(finite):
            depth = depth[finite]
        else:
            finite = np.ones((n,), dtype=bool)

        med = float(np.median(depth))
        abs_dev = np.abs(depth - med)
        mad = float(np.median(abs_dev))
        if mad > 1e-9:
            sigma = 1.4826 * mad
            keep_depth = abs_dev <= (3.0 * sigma)
            if np.any(keep_depth):
                keep = np.zeros((n,), dtype=bool)
                keep[np.where(finite)[0][keep_depth]] = True
                return keep

        q1 = float(np.percentile(depth, 25.0))
        q3 = float(np.percentile(depth, 75.0))
        iqr = q3 - q1
        if iqr > 1e-9:
            lo = q1 - 1.5 * iqr
            hi = q3 + 1.5 * iqr
            keep_depth = (depth >= lo) & (depth <= hi)
            if np.any(keep_depth):
                keep = np.zeros((n,), dtype=bool)
                keep[np.where(finite)[0][keep_depth]] = True
                return keep

        keep = np.zeros((n,), dtype=bool)
        keep[np.where(finite)[0]] = True
        return keep

    def _aggregate_patch_points(pts_xyz: np.ndarray) -> Optional[np.ndarray]:
        if pts_xyz.size == 0:
            return None
        pts = np.asarray(pts_xyz, dtype=np.float32).reshape(-1, 3)
        finite = np.all(np.isfinite(pts), axis=1)
        pts = pts[finite]
        if pts.shape[0] == 0:
            return None

        keep = _depth_inlier_mask(pts)
        if np.any(keep):
            pts = pts[keep]
        if pts.shape[0] == 0:
            return None

        med = np.median(pts, axis=0).astype(np.float64)
        n = int(pts.shape[0])
        if n >= 5:
            d = np.linalg.norm(pts.astype(np.float64) - med.reshape(1, 3), axis=1)
            keep_n = max(3, int(np.floor(0.8 * n)))
            keep_n = min(keep_n, n)
            idx = np.argpartition(d, keep_n - 1)[:keep_n]
            return np.mean(pts[idx], axis=0).astype(np.float32)
        return med.astype(np.float32)

    def _valid_with_aircraft_preference(
        valid_mask: np.ndarray,
        rr_min: int,
        rr_max: int,
        cc_min: int,
        cc_max: int,
    ) -> np.ndarray:
        if mask_aircraft is None:
            return valid_mask
        if mask_aircraft.shape[:2] != xyz_hw3.shape[:2]:
            return valid_mask
        sub_air = np.asarray(mask_aircraft[rr_min:rr_max, cc_min:cc_max], dtype=bool)
        if sub_air.shape != valid_mask.shape:
            return valid_mask
        preferred = valid_mask & sub_air
        if np.any(preferred):
            return preferred
        return valid_mask

    H, W, _ = xyz_hw3.shape
    r0 = int(np.clip(r0, 0, H - 1))
    c0 = int(np.clip(c0, 0, W - 1))

    rr_min = max(0, r0 - int(radius))
    rr_max = min(H, r0 + int(radius) + 1)
    cc_min = max(0, c0 - int(radius))
    cc_max = min(W, c0 + int(radius) + 1)
    sub = xyz_hw3[rr_min:rr_max, cc_min:cc_max, :]
    valid = np.all(np.isfinite(sub), axis=2)
    valid = _valid_with_aircraft_preference(valid, rr_min, rr_max, cc_min, cc_max)

    if not np.any(valid):
        if not bool(allow_global_fallback):
            return None
        # Fallback: find nearest finite xyz in the whole image (prefer aircraft pixels).
        valid_all = np.all(np.isfinite(xyz_hw3), axis=2)
        if mask_aircraft is not None and mask_aircraft.shape[:2] == xyz_hw3.shape[:2]:
            valid_pref = valid_all & np.asarray(mask_aircraft, dtype=bool)
            if np.any(valid_pref):
                valid_all = valid_pref
        if not np.any(valid_all):
            return None
        rr_all, cc_all = np.where(valid_all)
        d2_all = (
            (rr_all.astype(np.float64) - float(r0)) ** 2
            + (cc_all.astype(np.float64) - float(c0)) ** 2
        )
        gi = int(np.argmin(d2_all))
        return xyz_hw3[rr_all[gi], cc_all[gi], :].astype(np.float32)

    rr, cc = np.where(valid)
    rr_abs = rr + rr_min
    cc_abs = cc + cc_min
    pts_patch = xyz_hw3[rr_abs, cc_abs, :].reshape(-1, 3)
    p_est = _aggregate_patch_points(pts_patch)
    if p_est is not None:
        return p_est

    d2 = (rr_abs.astype(np.float64) - float(r0)) ** 2 + (cc_abs.astype(np.float64) - float(c0)) ** 2
    i = int(np.argmin(d2))
    return xyz_hw3[rr_abs[i], cc_abs[i], :].astype(np.float32)


def _show_open3d_bbox_and_keypoints(
    pts_all: np.ndarray,
    pts_bbox: np.ndarray,
    kp_xyz_by_name: List[Tuple[str, np.ndarray]],
    window_name: str,
) -> bool:
    try:
        import open3d as o3d
    except Exception as e:
        print(f"  [WARN] Open3D not available, skipping 3D view ({e})")
        return False

    geoms = []

    # Full scene cloud (gray)
    if pts_all.size > 0:
        pcd_all = o3d.geometry.PointCloud()
        pcd_all.points = o3d.utility.Vector3dVector(pts_all.astype(np.float64))
        pcd_all.paint_uniform_color([0.68, 0.68, 0.68])
        geoms.append(pcd_all)

    # Points inside predicted bbox (red)
    pcd_bbox = o3d.geometry.PointCloud()
    pcd_bbox.points = o3d.utility.Vector3dVector(pts_bbox.astype(np.float64))
    pcd_bbox.paint_uniform_color([1.0, 0.15, 0.15])
    geoms.append(pcd_bbox)

    if DRAW_3D_OBB and pts_bbox.shape[0] >= 10:
        try:
            obb = pcd_bbox.get_oriented_bounding_box()
            obb.color = (1.0, 0.0, 0.0)
            geoms.append(obb)
        except Exception:
            pass

    # Keypoints (blue spheres)
    for _, p in kp_xyz_by_name:
        sph = o3d.geometry.TriangleMesh.create_sphere(radius=float(DEBUG_KPT_SPHERE_RADIUS_M))
        sph.compute_vertex_normals()
        sph.paint_uniform_color([0.10, 0.45, 1.00])
        sph.translate(np.asarray(p, dtype=np.float64).reshape(3), relative=False)
        geoms.append(sph)

    o3d.visualization.draw_geometries(
        geoms,
        window_name=window_name,
        width=1280,
        height=720,
    )
    return True


def run(
    source: str,
    weights: str,
    out_dir: str,
    max_h5_files: Optional[int] = None,
    imgsz: int = 1024,
    conf: float = 0.25,
    device: str = "0",
    save_img: bool = False,
    yaml_kp_names: Optional[str] = None,
    kp_conf: float = 0.25,
    kp_patch_radius: int = 3,
    show_3d: bool = False,
    max_vis_scenes: int = 20,
    image_path: Optional[str] = None,
    image_dir: Optional[str] = None,
    print_kp_conf: bool = False,
    kp_conf_csv: Optional[str] = None,
    check_bbox_coverage: bool = False,
    bbox_full_thr: float = 0.995,
    bbox_cov_csv: Optional[str] = None,
) -> None:
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError(
            "ultralytics is required. Install with `pip install ultralytics`."
        ) from e

    weights_path = _resolve_weights(weights)
    print(f"[model] Using weights: {weights_path}")

    model = YOLO(str(weights_path))
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    img_root = out_root / "debug_imgs"
    keypoint_names = _load_keypoint_names_from_yaml(yaml_kp_names)
    if save_img:
        img_root.mkdir(parents=True, exist_ok=True)
        if yaml_kp_names and keypoint_names:
            print(f"[kpts] Loaded {len(keypoint_names)} keypoint names from: {yaml_kp_names}")
        elif yaml_kp_names and not keypoint_names:
            print(f"[kpts] Could not load keypoint names from: {yaml_kp_names} (using K0, K1, ...)")

    print("[list] Searching for .h5 files…")
    h5_paths = list_h5_paths(source)
    if not h5_paths:
        raise RuntimeError(f"No .h5 files found under: {source}")

    if image_path and image_dir:
        raise RuntimeError("Use only one of --image-path or --image-dir, not both.")

    image_mode = bool(image_path or image_dir)
    scene_requests_by_h5: Dict[str, List[Dict[str, Any]]] = {}
    if image_mode:
        by_stem: Dict[str, List[str]] = {}
        for p in h5_paths:
            by_stem.setdefault(Path(p).stem, []).append(p)

        if image_path:
            image_paths = [_resolve_image_path_with_split_fallback(image_path)]
            print(f"[image-mode] Single image requested: {image_paths[0]}")
        else:
            image_dir_path = Path(str(image_dir)).expanduser().resolve()
            image_paths = _collect_images_from_dir(image_dir_path)
            print(f"[image-mode] Using all images in: {image_dir_path} (count={len(image_paths)})")

        skipped_bad_stem = 0
        skipped_missing_h5 = 0
        skipped_unreadable = 0

        for ip in image_paths:
            img_bgr = cv2.imread(str(ip))
            if img_bgr is None:
                skipped_unreadable += 1
                print(f"[image-mode][skip] unreadable image: {ip}")
                continue
            try:
                h5_stem, scene_name = _parse_unique_scene_stem(ip.stem)
            except Exception as e:
                skipped_bad_stem += 1
                print(f"[image-mode][skip] bad image stem: {ip.name} ({e})")
                continue

            matches = by_stem.get(h5_stem, [])
            if not matches:
                skipped_missing_h5 += 1
                print(f"[image-mode][skip] no H5 match for stem='{h5_stem}' image={ip.name}")
                continue
            if len(matches) > 1:
                print(
                    f"[image-mode][warn] Multiple H5 files match stem '{h5_stem}'. "
                    f"Using first: {matches[0]}"
                )
            h5_match = matches[0]
            scene_requests_by_h5.setdefault(h5_match, []).append(
                {
                    "scene_name": str(scene_name),
                    "image_path": ip.resolve(),
                    "image_bgr": img_bgr,
                }
            )

        if not scene_requests_by_h5:
            raise RuntimeError(
                "No valid image->H5 scene mapping found. "
                "Check image filenames '<h5_stem>__<scene_name>.png' and --source."
            )

        h5_paths = list(scene_requests_by_h5.keys())
        total_requests = sum(len(v) for v in scene_requests_by_h5.values())
        print(
            "[image-mode] mapped requests: "
            f"{total_requests} scenes across {len(h5_paths)} H5 files "
            f"(skipped: unreadable={skipped_unreadable}, bad_stem={skipped_bad_stem}, missing_h5={skipped_missing_h5})"
        )

    if (not image_mode) and max_h5_files is not None and max_h5_files > 0 and len(h5_paths) > max_h5_files:
        h5_paths = h5_paths[:max_h5_files]
        print(f"[list] Using only first {len(h5_paths)} HDF5 files (max_h5_files={max_h5_files})")

    total_scenes = 0
    total_saved = 0
    total_img_saved = 0
    total_3d_shown = 0
    kp_conf_vals: Dict[int, List[float]] = defaultdict(list)
    kp_conf_scene_rows = 0
    kp_conf_rows: List[Dict[str, Any]] = []
    bbox_cov_rows: List[Dict[str, Any]] = []
    bbox_cov_pass = 0
    bbox_cov_fail = 0
    bbox_cov_with_mask = 0
    bbox_cov_reason_counts: Dict[str, int] = defaultdict(int)

    def _record_bbox_result(
        *,
        unique_scene_name: str,
        h5_file_name: str,
        scene_name_str: str,
        bbox_xyxy: Optional[Tuple[int, int, int, int]],
        status: str,
        reason: str,
        aircraft_px_total: Optional[int] = None,
        aircraft_px_inside: Optional[int] = None,
        aircraft_recall: Optional[float] = None,
        bbox_area_px: Optional[int] = None,
    ) -> None:
        nonlocal bbox_cov_pass, bbox_cov_fail, bbox_cov_with_mask
        x1v: Optional[int] = None
        y1v: Optional[int] = None
        x2v: Optional[int] = None
        y2v: Optional[int] = None
        if bbox_xyxy is not None:
            x1v, y1v, x2v, y2v = [int(v) for v in bbox_xyxy]
        row = {
            "unique_scene": unique_scene_name,
            "h5_file": h5_file_name,
            "scene_name": scene_name_str,
            "bbox_x1": x1v,
            "bbox_y1": y1v,
            "bbox_x2": x2v,
            "bbox_y2": y2v,
            "aircraft_px_total": aircraft_px_total,
            "aircraft_px_inside": aircraft_px_inside,
            "aircraft_recall": aircraft_recall,
            "bbox_area_px": bbox_area_px,
            "bbox_status": str(status).upper(),
            "bbox_reason": str(reason),
            "full_threshold": float(bbox_full_thr),
        }
        bbox_cov_rows.append(row)
        if row["bbox_status"] == "PASS":
            bbox_cov_pass += 1
        else:
            bbox_cov_fail += 1
        if aircraft_recall is not None:
            bbox_cov_with_mask += 1
        bbox_cov_reason_counts[str(reason)] += 1

    for i, h5p in enumerate(h5_paths, 1):
        print(f"[{i}/{len(h5_paths)}] {Path(h5p).name}")
        try:
            with open_h5_any(h5p) as f:
                H = int(f.attrs["height"])
                W = int(f.attrs["width"])

                if image_mode:
                    scene_reqs = scene_requests_by_h5.get(h5p, [])
                    scene_iter = [
                        (str(req["scene_name"]), req.get("image_path"), req.get("image_bgr"))
                        for req in scene_reqs
                    ]
                else:
                    scene_iter = [
                        (s, None, None)
                        for s, g in f.items()
                        if isinstance(g, h5py.Group) and "points" in g
                    ]

                for scene_name, scene_image_path, scene_image_bgr in scene_iter:
                    if scene_name not in f:
                        print(f"  [SKIP] Scene not found in H5: {scene_name}")
                        continue
                    grp = f[scene_name]
                    if not isinstance(grp, h5py.Group) or "points" not in grp:
                        print(f"  [SKIP] Scene has no points group: {scene_name}")
                        continue

                    total_scenes += 1
                    file_stem = Path(h5p).stem
                    unique_scene = f"{file_stem}__{scene_name}"

                    ds = grp["points"]
                    flat = ds[()]  # (H*W, C)

                    cols_raw = ds.attrs.get("columns", None)
                    cols = [
                        c.decode("utf-8") if isinstance(c, (bytes, bytearray)) else str(c)
                        for c in (cols_raw if cols_raw is not None else [])
                    ]
                    if not cols:
                        cols = [
                            "x",
                            "y",
                            "z",
                            "range",
                            "intensity",
                            "reflectivity",
                            "ambient",
                            "is_ground",
                            "is_aircraft",
                        ]

                    rgb, xyz_hw3 = _build_rgb_and_xyz(flat, cols, H, W)
                    if rgb is None or xyz_hw3 is None:
                        print(f"  [SKIP] Missing required cols for {unique_scene}")
                        if check_bbox_coverage:
                            _record_bbox_result(
                                unique_scene_name=unique_scene,
                                h5_file_name=Path(h5p).name,
                                scene_name_str=str(scene_name),
                                bbox_xyxy=None,
                                status="FAIL",
                                reason="missing_required_cols",
                            )
                            print(
                                f"  [bbox-cover] {unique_scene}: "
                                "status=FAIL reason=missing_required_cols"
                            )
                        continue

                    mask_aircraft_eval = None
                    if check_bbox_coverage or scene_image_bgr is not None:
                        mask_aircraft_eval = _extract_is_aircraft_mask(flat, cols, H, W)

                    # In image-mode we run YOLO on an already exported dataset image.
                    # Export may have azimuth-roll applied, so we must roll xyz the
                    # same way before bbox/keypoint backprojection.
                    if scene_image_bgr is not None and mask_aircraft_eval is not None:
                        shift_cols = _compute_export_like_roll(mask_aircraft_eval)
                        if shift_cols != 0:
                            xyz_hw3 = np.roll(xyz_hw3, shift=shift_cols, axis=1)
                            rgb = np.roll(rgb, shift=shift_cols, axis=1)
                            mask_aircraft_eval = np.roll(mask_aircraft_eval, shift=shift_cols, axis=1)
                            print(
                                f"  [image-mode] Applied azimuth roll to xyz/rgb: "
                                f"{shift_cols} cols"
                            )

                    rgb_for_model = rgb
                    if scene_image_bgr is not None:
                        rgb_for_model = scene_image_bgr
                        if rgb_for_model.shape[0] != H or rgb_for_model.shape[1] != W:
                            print(
                                f"  [image-mode] Resizing image from "
                                f"{rgb_for_model.shape[1]}x{rgb_for_model.shape[0]} to {W}x{H}"
                            )
                            rgb_for_model = cv2.resize(rgb_for_model, (W, H), interpolation=cv2.INTER_LINEAR)

                    # YOLO inference
                    results = model.predict(rgb_for_model, imgsz=imgsz, conf=conf, device=device, verbose=False)
                    if not results:
                        print(f"  [SKIP] No results for {unique_scene}")
                        if check_bbox_coverage:
                            _record_bbox_result(
                                unique_scene_name=unique_scene,
                                h5_file_name=Path(h5p).name,
                                scene_name_str=str(scene_name),
                                bbox_xyxy=None,
                                status="FAIL",
                                reason="no_yolo_result",
                            )
                            print(
                                f"  [bbox-cover] {unique_scene}: "
                                "status=FAIL reason=no_yolo_result"
                            )
                        continue

                    r0 = results[0]
                    det_i = _pick_best_det_idx(r0, target_class=0)
                    bb = None
                    if det_i is not None and r0.boxes is not None and len(r0.boxes) > det_i:
                        x1f, y1f, x2f, y2f = r0.boxes.xyxy.detach().cpu().numpy()[det_i].tolist()
                        bb = (int(x1f), int(y1f), int(x2f), int(y2f))
                    if bb is None:
                        bb = _pick_best_bbox(r0, target_class=0)
                    if bb is None:
                        print(f"  [SKIP] No aircraft bbox for {unique_scene}")
                        if check_bbox_coverage:
                            _record_bbox_result(
                                unique_scene_name=unique_scene,
                                h5_file_name=Path(h5p).name,
                                scene_name_str=str(scene_name),
                                bbox_xyxy=None,
                                status="FAIL",
                                reason="no_bbox",
                            )
                            print(
                                f"  [bbox-cover] {unique_scene}: "
                                "status=FAIL reason=no_bbox"
                            )
                        continue

                    x1, y1, x2, y2 = bb
                    x1 = int(np.clip(x1, 0, W - 1))
                    x2 = int(np.clip(x2, 0, W - 1))
                    y1 = int(np.clip(y1, 0, H - 1))
                    y2 = int(np.clip(y2, 0, H - 1))
                    if x2 < x1 or y2 < y1:
                        print(f"  [SKIP] Invalid bbox for {unique_scene}")
                        if check_bbox_coverage:
                            _record_bbox_result(
                                unique_scene_name=unique_scene,
                                h5_file_name=Path(h5p).name,
                                scene_name_str=str(scene_name),
                                bbox_xyxy=(x1, y1, x2, y2),
                                status="FAIL",
                                reason="invalid_bbox",
                            )
                            print(
                                f"  [bbox-cover] {unique_scene}: "
                                f"bbox=({x1},{y1},{x2},{y2}) status=FAIL reason=invalid_bbox"
                            )
                        continue

                    if check_bbox_coverage:
                        cov = _bbox_aircraft_coverage(mask_aircraft_eval, (x1, y1, x2, y2))
                        if cov is None:
                            _record_bbox_result(
                                unique_scene_name=unique_scene,
                                h5_file_name=Path(h5p).name,
                                scene_name_str=str(scene_name),
                                bbox_xyxy=(x1, y1, x2, y2),
                                status="FAIL",
                                reason="missing_aircraft_mask",
                                bbox_area_px=int((x2 - x1 + 1) * (y2 - y1 + 1)),
                            )
                            print(
                                f"  [bbox-cover] {unique_scene}: "
                                f"bbox=({x1},{y1},{x2},{y2}) status=FAIL reason=missing_aircraft_mask"
                            )
                        else:
                            rec = float(cov["aircraft_recall"])
                            inside = int(round(float(cov["aircraft_px_inside"])))
                            total = int(round(float(cov["aircraft_px_total"])))
                            is_pass = rec >= float(bbox_full_thr)
                            reason = "full_aircraft_covered" if is_pass else "aircraft_outside_bbox"
                            _record_bbox_result(
                                unique_scene_name=unique_scene,
                                h5_file_name=Path(h5p).name,
                                scene_name_str=str(scene_name),
                                bbox_xyxy=(x1, y1, x2, y2),
                                status=("PASS" if is_pass else "FAIL"),
                                reason=reason,
                                aircraft_px_total=total,
                                aircraft_px_inside=inside,
                                aircraft_recall=rec,
                                bbox_area_px=int(round(float(cov["bbox_area_px"]))),
                            )
                            print(
                                f"  [bbox-cover] {unique_scene}: "
                                f"bbox=({x1},{y1},{x2},{y2}) "
                                f"inside={inside}/{total} ({rec * 100.0:.2f}%) "
                                f"status={'PASS' if is_pass else 'FAIL'} "
                                f"thr={float(bbox_full_thr):.3f}"
                            )

                    # crop points to bbox
                    pts = xyz_hw3[y1 : y2 + 1, x1 : x2 + 1, :].reshape(-1, 3)
                    finite = np.all(np.isfinite(pts), axis=1)
                    pts = pts[finite]
                    if pts.size == 0:
                        print(f"  [SKIP] Empty pointcloud for {unique_scene}")
                        continue
                    pts_all = xyz_hw3.reshape(-1, 3)
                    pts_all = pts_all[np.all(np.isfinite(pts_all), axis=1)]

                    kp_xy = None
                    kp_conf_arr = None
                    kp_xyz_by_name: List[Tuple[str, np.ndarray]] = []
                    if det_i is not None and getattr(r0, "keypoints", None) is not None and r0.keypoints is not None:
                        if r0.keypoints.xy is not None:
                            all_xy = r0.keypoints.xy.detach().cpu().numpy()
                            if all_xy.ndim == 3 and det_i < all_xy.shape[0]:
                                kp_xy = all_xy[det_i]
                        if getattr(r0.keypoints, "conf", None) is not None and r0.keypoints.conf is not None:
                            all_kc = r0.keypoints.conf.detach().cpu().numpy()
                            if all_kc.ndim == 2 and det_i < all_kc.shape[0]:
                                kp_conf_arr = all_kc[det_i]

                    if print_kp_conf:
                        if kp_conf_arr is None:
                            print(f"  [kpt-conf] {unique_scene}: unavailable")
                            kp_conf_rows.append(
                                {
                                    "unique_scene": unique_scene,
                                    "h5_file": Path(h5p).name,
                                    "scene_name": str(scene_name),
                                    "kp_conf": [],
                                    "has_conf": False,
                                }
                            )
                        else:
                            parts: List[str] = []
                            conf_list: List[float] = []
                            for k, cv in enumerate(kp_conf_arr):
                                cval = float(cv)
                                kp_name = keypoint_names[k] if k < len(keypoint_names) else f"K{k}"
                                parts.append(f"{kp_name}={cval:.3f}")
                                kp_conf_vals[k].append(cval)
                                conf_list.append(cval)
                            kp_conf_rows.append(
                                {
                                    "unique_scene": unique_scene,
                                    "h5_file": Path(h5p).name,
                                    "scene_name": str(scene_name),
                                    "kp_conf": conf_list,
                                    "has_conf": True,
                                }
                            )
                            kp_conf_scene_rows += 1
                            print(f"  [kpt-conf] {unique_scene}: " + "  ".join(parts))

                    if kp_xy is not None:
                        for k, (x, y) in enumerate(kp_xy):
                            if kp_conf_arr is not None and k < len(kp_conf_arr):
                                if float(kp_conf_arr[k]) < float(kp_conf):
                                    continue
                            rk = int(round(float(y)))
                            ck = int(round(float(x)))
                            p3 = _sample_xyz_nearest(
                                xyz_hw3=xyz_hw3,
                                r0=rk,
                                c0=ck,
                                radius=kp_patch_radius,
                                allow_global_fallback=ALLOW_GLOBAL_KP_XYZ_FALLBACK,
                                mask_aircraft=mask_aircraft_eval,
                            )
                            if p3 is None:
                                continue
                            kp_name = keypoint_names[k] if k < len(keypoint_names) else f"K{k}"
                            kp_xyz_by_name.append((kp_name, p3))

                    pts_to_save = pts
                    if APPEND_KEYPOINTS_TO_PCD and kp_xyz_by_name:
                        kp_pts = np.stack([p for _, p in kp_xyz_by_name], axis=0).astype(np.float32)
                        pts_to_save = np.concatenate([pts, kp_pts], axis=0)

                    out_path = out_root / f"{unique_scene}.pcd"
                    write_pcd_xyz(out_path, pts_to_save)
                    total_saved += 1

                    if save_img:
                        path_lines = None
                        if DRAW_PATH_TEXT_ON_IMAGE:
                            path_lines = [
                                f"scene: {unique_scene}",
                                f"pcd: {out_path.name}",
                            ]
                            if scene_image_path is not None:
                                path_lines.append(f"img: {Path(scene_image_path).name}")
                        dbg = _draw_bbox_and_keypoints(
                            rgb=rgb_for_model,
                            bb=(x1, y1, x2, y2),
                            kp_xy=kp_xy,
                            kp_conf_arr=kp_conf_arr,
                            kp_conf_thr=kp_conf,
                            kp_names=keypoint_names,
                            draw_bbox=bool(DRAW_BBOX_ON_DEBUG_IMAGE),
                            path_lines=path_lines,
                        )
                        cv2.imwrite(str(img_root / f"{unique_scene}.png"), dbg)
                        total_img_saved += 1

                    if show_3d and total_3d_shown < int(max_vis_scenes):
                        ok = _show_open3d_bbox_and_keypoints(
                            pts_all=pts_all,
                            pts_bbox=pts,
                            kp_xyz_by_name=kp_xyz_by_name,
                            window_name=f"YOLO->PCD {unique_scene}",
                        )
                        if ok:
                            total_3d_shown += 1

        except Exception as e:
            print(f"[ERROR] {Path(h5p).name}: {e}")

    print(f"\n[summary] scenes processed: {total_scenes}")
    print(f"[summary] pcd saved: {total_saved}")
    if save_img:
        print(f"[summary] debug images saved (bbox + keypoints): {total_img_saved}")
        print("  -> debug images:", img_root.resolve())
    if show_3d:
        print(f"[summary] 3D windows shown: {total_3d_shown}")
    if print_kp_conf:
        if kp_conf_scene_rows == 0:
            print("[summary] keypoint confidence: no confidence arrays available")
        else:
            print("[summary] keypoint confidence stats:")
            for k in sorted(kp_conf_vals.keys()):
                arr = np.asarray(kp_conf_vals[k], dtype=np.float32)
                if arr.size == 0:
                    continue
                kp_name = keypoint_names[k] if k < len(keypoint_names) else f"K{k}"
                print(
                    f"  {kp_name}: n={arr.size} "
                    f"mean={float(arr.mean()):.3f} min={float(arr.min()):.3f} max={float(arr.max()):.3f}"
                )
        if kp_conf_rows:
            csv_path = (
                Path(str(kp_conf_csv)).expanduser().resolve()
                if kp_conf_csv and str(kp_conf_csv).strip()
                else (out_root / "keypoint_confidence.csv").resolve()
            )
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            max_k = max(len(r.get("kp_conf", [])) for r in kp_conf_rows)
            if max_k <= 0:
                max_k = len(keypoint_names)
            conf_headers = [
                (
                    keypoint_names[k]
                    if k < len(keypoint_names) and str(keypoint_names[k]).strip()
                    else f"K{k}"
                )
                for k in range(max_k)
            ]
            with csv_path.open("w", newline="", encoding="utf-8") as f_csv:
                writer = csv.writer(f_csv)
                writer.writerow(
                    [
                        "unique_scene",
                        "h5_file",
                        "scene_name",
                        "has_conf",
                        "num_conf",
                    ]
                    + [f"conf_{name}" for name in conf_headers]
                )
                for row in kp_conf_rows:
                    vals = list(row.get("kp_conf", []))
                    padded_vals = [
                        f"{float(vals[i]):.6f}" if i < len(vals) else ""
                        for i in range(max_k)
                    ]
                    writer.writerow(
                        [
                            row.get("unique_scene", ""),
                            row.get("h5_file", ""),
                            row.get("scene_name", ""),
                            1 if bool(row.get("has_conf", False)) else 0,
                            len(vals),
                        ]
                        + padded_vals
                    )
            print(f"[summary] keypoint confidence CSV: {csv_path}")
    if check_bbox_coverage:
        total_bbox_rows = len(bbox_cov_rows)
        if total_bbox_rows <= 0:
            print("[summary] bbox pass/fail: no rows recorded")
        else:
            rate = 100.0 * float(bbox_cov_pass) / float(total_bbox_rows)
            print(
                f"[summary] bbox pass/fail: scenes={total_bbox_rows} "
                f"pass={bbox_cov_pass} fail={bbox_cov_fail} ({rate:.2f}% pass)"
            )
            print(
                f"[summary] bbox full-aircraft (mask-based): checked={bbox_cov_with_mask} "
                f"thr={float(bbox_full_thr):.3f}"
            )
            if bbox_cov_reason_counts:
                for reason, count in sorted(bbox_cov_reason_counts.items(), key=lambda kv: (-kv[1], kv[0])):
                    print(f"[summary] bbox reason: {reason} -> {count}")
        if bbox_cov_rows:
            cov_csv_path = (
                Path(str(bbox_cov_csv)).expanduser().resolve()
                if bbox_cov_csv and str(bbox_cov_csv).strip()
                else (out_root / "bbox_aircraft_coverage.csv").resolve()
            )
            cov_csv_path.parent.mkdir(parents=True, exist_ok=True)
            with cov_csv_path.open("w", newline="", encoding="utf-8") as f_csv:
                writer = csv.writer(f_csv)
                writer.writerow(
                    [
                        "unique_scene",
                        "h5_file",
                        "scene_name",
                        "bbox_x1",
                        "bbox_y1",
                        "bbox_x2",
                        "bbox_y2",
                        "bbox_status",
                        "bbox_reason",
                        "aircraft_px_total",
                        "aircraft_px_inside",
                        "aircraft_recall",
                        "bbox_area_px",
                        "full_threshold",
                    ]
                )
                for row in bbox_cov_rows:
                    rec_v = row.get("aircraft_recall", None)
                    writer.writerow(
                        [
                            row.get("unique_scene", ""),
                            row.get("h5_file", ""),
                            row.get("scene_name", ""),
                            "" if row.get("bbox_x1", None) is None else int(row.get("bbox_x1")),
                            "" if row.get("bbox_y1", None) is None else int(row.get("bbox_y1")),
                            "" if row.get("bbox_x2", None) is None else int(row.get("bbox_x2")),
                            "" if row.get("bbox_y2", None) is None else int(row.get("bbox_y2")),
                            row.get("bbox_status", ""),
                            row.get("bbox_reason", ""),
                            "" if row.get("aircraft_px_total", None) is None else int(row.get("aircraft_px_total")),
                            "" if row.get("aircraft_px_inside", None) is None else int(row.get("aircraft_px_inside")),
                            "" if rec_v is None else f"{float(rec_v):.6f}",
                            "" if row.get("bbox_area_px", None) is None else int(row.get("bbox_area_px")),
                            f"{float(row.get('full_threshold', 0.0)):.6f}",
                        ]
                    )
            print(f"[summary] bbox coverage CSV: {cov_csv_path}")
    print("  -> output:", out_root.resolve())


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Run YOLO-Pose .pt on H5 range images, save bbox-cropped pointclouds as .pcd, "
            "and optionally visualize bbox+keypoints on image and in 3D."
        )
    )
    p.add_argument("--source", type=str, default=SOURCE, help="Input H5 directory or gs:// path")
    p.add_argument(
        "--image-path",
        type=str,
        default=None,
        help=(
            "Optional single test image path '<h5_stem>__<scene_name>.<ext>'. "
            "If set, script finds matching H5+scene and backprojects keypoints/bbox to PCD."
        ),
    )
    p.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help=(
            "Optional directory of test images. If set, all images in this folder are "
            "mapped by filename stem '<h5_stem>__<scene_name>' and exported to PCD."
        ),
    )
    p.add_argument(
        "--weights",
        type=str,
        default="/home/femi/yolo_pose_dataset_creation/runs/pose/aircraft_pose_exp_after_point_clound_check/weights/best.pt",
        help="YOLO .pt weights file or a directory containing best.pt/last.pt",
    )
    p.add_argument("--out", type=str, default="./pcd_from_yolo", help="Output directory for .pcd files")
    p.add_argument("--max-h5", type=int, default=None, help="Limit number of H5 files")
    p.add_argument("--imgsz", type=int, default=1024, help="YOLO inference image size")
    p.add_argument("--conf", type=float, default=0.05, help="YOLO confidence threshold")
    p.add_argument("--device", type=str, default="0", help="YOLO device (e.g. 0, 0,1, or cpu)")
    p.add_argument(
        "--save-img",
        action="store_true",
        help="Save debug RGB images with bbox and keypoint overlay",
    )
    p.add_argument(
        "--yaml-kp-names",
        type=str,
        default=DEFAULT_YAML_KP_NAMES,
        help="Optional aircraft_pose.yaml path to label keypoints on debug image",
    )
    p.add_argument(
        "--kp-conf",
        type=float,
        default=0.9,
        help="Keypoint confidence threshold for drawing / 3D backprojection",
    )
    p.add_argument(
        "--kp-patch-radius",
        type=int,
        default=3,
        help="Patch radius for finding nearest valid xyz around each keypoint pixel",
    )
    p.add_argument(
        "--show-3d",
        action="store_true",
        help="Show Open3D window with bbox-cropped pointcloud and keypoint markers",
    )
    p.add_argument(
        "--max-vis-scenes",
        type=int,
        default=20,
        help="Maximum number of scenes to open in 3D viewer",
    )
    p.add_argument(
        "--print-kp-conf",
        action="store_true",
        help="Print per-scene keypoint confidence and end-of-run confidence stats",
    )
    p.add_argument(
        "--kp-conf-csv",
        type=str,
        default="",
        help="Optional output CSV path for keypoint confidences (default: <out>/keypoint_confidence.csv)",
    )
    p.add_argument(
        "--check-bbox-coverage",
        action="store_true",
        help="Check whether predicted bbox covers full aircraft mask and print per-scene stats",
    )
    p.add_argument(
        "--bbox-full-thr",
        type=float,
        default=0.995,
        help="Coverage threshold on aircraft pixels inside bbox to mark full-aircraft inclusion",
    )
    p.add_argument(
        "--bbox-cov-csv",
        type=str,
        default="",
        help="Optional CSV output path for bbox coverage (default: <out>/bbox_aircraft_coverage.csv)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if USE_CODE_PATH_OVERRIDES:
        source = str(CODE_SOURCE_PATH)
        weights = str(CODE_WEIGHTS_PATH)
        out_dir = str(CODE_OUT_DIR)
        image_path = CODE_IMAGE_PATH
        image_dir = CODE_IMAGE_DIR
        yaml_kp_names = str(CODE_YAML_KP_NAMES)
        kp_conf_csv = "" if CODE_KP_CONF_CSV is None else str(CODE_KP_CONF_CSV)
        print("[config] Using code-level path overrides from test_yolo_pose_from_h5_weights_to_pcd.py")
    else:
        source = args.source
        weights = args.weights
        out_dir = args.out
        image_path = args.image_path
        image_dir = args.image_dir
        yaml_kp_names = args.yaml_kp_names
        kp_conf_csv = args.kp_conf_csv

    run(
        source=source,
        weights=weights,
        out_dir=out_dir,
        max_h5_files=args.max_h5,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        save_img=args.save_img,
        yaml_kp_names=yaml_kp_names,
        kp_conf=args.kp_conf,
        kp_patch_radius=args.kp_patch_radius,
        show_3d=args.show_3d,
        max_vis_scenes=args.max_vis_scenes,
        image_path=image_path,
        image_dir=image_dir,
        print_kp_conf=args.print_kp_conf,
        kp_conf_csv=kp_conf_csv,
        check_bbox_coverage=args.check_bbox_coverage,
        bbox_full_thr=args.bbox_full_thr,
        bbox_cov_csv=args.bbox_cov_csv,
    )
