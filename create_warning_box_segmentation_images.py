#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create segmentation masks from warning-box 3D inclusion.

For each input image (or images in a directory), this script:
1) Parses unique scene id from filename stem: <h5_stem>__<scene_name>
2) Loads H5 scene XYZ (H, W, 3) with the same export-like roll used elsewhere
3) Builds warning boxes in world coordinates using existing warning-profile/YAML logic
   - optional A380 override: front_landing_gear spec from YOLO detection bbox
4) Marks pixels whose XYZ lies inside each warning box
5) Saves:
   - class-id mask PNG (uint8)
   - optional color overlay PNG
   - CSV summary

Class IDs in output mask are configured via --class-map.
Default class map:
  front_landing_gear:1,engine_left:2,engine_right:3
Background is always 0.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import h5py
import numpy as np

from io_helpers import list_h5_paths, open_h5_any


DEFAULT_IMAGE_PATH = (
    "/home/femi/yolo_pose_dataset_creation/aircraft_engine_seg_with_front_auto/images"
)
DEFAULT_SOURCE_H5_ROOT = "/home/femi/Benchmarking_framework/Data/warning_b_test_h5"
DEFAULT_OUT_DIR = "/home/femi/yolo_pose_dataset_creation/warning_box_seg_masks"
DEFAULT_CLASS_MAP = "front_landing_gear:1,engine_left:2,engine_right:3"
DEFAULT_YOLO_YAML_NAME = "aircraft_warning_box_seg.yaml"
DEFAULT_A380_DET_DATASET_ROOT = "/home/femi/yolo_pose_dataset_creation/aircraft_engine_det_with_front"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class SceneRequest:
    unique_scene: str
    image_path: Path
    image_bgr: np.ndarray


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Create segmentation masks from warning-box 3D inclusion."
    )
    ap.add_argument(
        "--image-path",
        type=str,
        default=DEFAULT_IMAGE_PATH,
        help="Single image path or directory path",
    )
    ap.add_argument(
        "--image-dir",
        type=str,
        default="",
        help="Directory containing input images (alternative to --image-path)",
    )
    ap.add_argument(
        "--source",
        type=str,
        default=DEFAULT_SOURCE_H5_ROOT,
        help="Root folder containing source H5 files",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=DEFAULT_OUT_DIR,
        help="Output directory",
    )
    ap.add_argument(
        "--class-map",
        type=str,
        default=DEFAULT_CLASS_MAP,
        help=(
            "Comma-separated mapping of warning-box key to class id, e.g. "
            "'front_landing_gear:1,engine_left:2,engine_right:3'"
        ),
    )
    ap.add_argument(
        "--warning-box-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to warning-box sizes",
    )
    ap.add_argument(
        "--warning-profile-csv",
        type=str,
        default="",
        help="Optional warning profile CSV path (defaults to view_pcd_dir constant)",
    )
    ap.add_argument(
        "--warning-yaml-column",
        type=str,
        default="recommended_yaml",
        help="CSV column name with warning YAML path",
    )
    ap.add_argument(
        "--warning-yaml-root",
        type=str,
        default="",
        help="Warning YAML root directory (fallback resolver)",
    )
    ap.add_argument(
        "--warning-yaml-relpath",
        type=str,
        default="detection_configs/default.yaml",
        help="Relative warning YAML path under aircraft folder",
    )
    ap.add_argument(
        "--warning-target-level",
        type=int,
        default=5,
        help="Warning level filter (falls back to all if unavailable)",
    )
    ap.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Maximum images to process (0 = all)",
    )
    ap.add_argument(
        "--save-overlay",
        type=int,
        default=1,
        choices=[0, 1],
        help="Save color overlay image",
    )
    ap.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.45,
        help="Overlay alpha in [0,1]",
    )
    ap.add_argument(
        "--include-aircraft-mask",
        type=int,
        default=1,
        choices=[0, 1],
        help="Also write aircraft mask from H5 is_aircraft field",
    )
    ap.add_argument(
        "--aircraft-class-id",
        type=int,
        default=4,
        help="Class id used for aircraft mask when --include-aircraft-mask=1",
    )
    ap.add_argument(
        "--edits-json",
        type=str,
        default="",
        help=(
            "Optional JSON file containing per-scene warning-box edits. "
            "Edits override warning specs before mask generation."
        ),
    )
    ap.add_argument(
        "--export-yolo-seg",
        type=int,
        default=1,
        choices=[0, 1],
        help="Also export YOLO segmentation dataset (images/labels).",
    )
    ap.add_argument(
        "--yolo-dataset-dir",
        type=str,
        default="",
        help="Output folder for YOLO dataset (default: <out>/yolo_seg_dataset).",
    )
    ap.add_argument(
        "--yolo-split",
        type=str,
        default="auto",
        choices=["auto", "train", "val", "test"],
        help="Dataset split used for YOLO export. 'auto' infers from image path.",
    )
    ap.add_argument(
        "--yolo-yaml-name",
        type=str,
        default=DEFAULT_YOLO_YAML_NAME,
        help="Filename of YOLO dataset yaml written inside yolo-dataset-dir.",
    )
    ap.add_argument(
        "--min-contour-area",
        type=float,
        default=25.0,
        help="Minimum contour area in pixels for one YOLO polygon instance.",
    )
    ap.add_argument(
        "--contour-approx-eps",
        type=float,
        default=1.0,
        help="Polygon simplification epsilon in pixels (0 to disable).",
    )
    ap.add_argument(
        "--a380-front-from-det-bbox",
        type=int,
        default=1,
        choices=[0, 1],
        help="For a380_800 scenes, replace front_landing_gear YAML spec using YOLO detection bbox points.",
    )
    ap.add_argument(
        "--a380-det-dataset-root",
        type=str,
        default=DEFAULT_A380_DET_DATASET_ROOT,
        help="YOLO detection dataset root containing labels/train|val|test/*.txt.",
    )
    ap.add_argument(
        "--a380-det-front-class-id",
        type=int,
        default=3,
        help="YOLO detection class id for front_gear bbox (default 3 in aircraft_engine_det_with_front).",
    )
    ap.add_argument(
        "--a380-det-split",
        type=str,
        default="auto",
        choices=["auto", "train", "val", "test"],
        help="Split used to read YOLO detection labels for A380 override. 'auto' infers from image path.",
    )
    ap.add_argument(
        "--a380-det-half-expand",
        type=float,
        default=1.0,
        help="Scale factor applied to fitted front_landing_gear half-size from detection bbox points.",
    )
    ap.add_argument(
        "--a380-det-min-points",
        type=int,
        default=30,
        help="Minimum valid XYZ points required in front bbox to apply A380 override.",
    )
    return ap.parse_args()


def _collect_images_from_dir(image_dir: Path) -> List[Path]:
    out: List[Path] = []
    if not image_dir.exists() or not image_dir.is_dir():
        return out
    for p in sorted(image_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            out.append(p.resolve())
    return out


def _parse_unique_scene_stem(stem: str) -> Tuple[str, str]:
    s = str(stem).strip()
    if "__" not in s:
        raise ValueError(
            f"Image stem '{s}' missing '__'. Expected '<h5_stem>__<scene_name>'."
        )
    h5_stem, scene_name = s.rsplit("__", 1)
    if not h5_stem or not scene_name:
        raise ValueError(f"Invalid scene stem: '{s}'")
    return h5_stem, scene_name


def _build_h5_index(source_root: str) -> Dict[str, List[Path]]:
    by_stem: Dict[str, List[Path]] = defaultdict(list)
    for hp in list_h5_paths(str(source_root)):
        p = Path(hp).expanduser().resolve()
        by_stem[str(p.stem)].append(p)
    return by_stem


def _load_scene_xyz(
    *,
    h5_path: Path,
    scene_name: str,
    pose_mod: Any,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int, int, str]:
    try:
        with open_h5_any(str(h5_path)) as f:
            H = int(f.attrs["height"])
            W = int(f.attrs["width"])
            if scene_name not in f:
                return None, None, 0, 0, f"scene_missing:{scene_name}"
            grp = f[scene_name]
            if not isinstance(grp, h5py.Group) or "points" not in grp:
                return None, None, 0, 0, "points_missing"
            ds = grp["points"]
            flat = ds[()]
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

            _rgb, xyz_hw3 = pose_mod._build_rgb_and_xyz(flat, cols, H, W)
            if xyz_hw3 is None:
                return None, None, 0, 0, "xyz_build_failed"

            mask_aircraft = pose_mod._extract_is_aircraft_mask(flat, cols, H, W)
            if mask_aircraft is not None:
                shift_cols = int(pose_mod._compute_export_like_roll(mask_aircraft))
                if shift_cols != 0:
                    xyz_hw3 = np.roll(xyz_hw3, shift=shift_cols, axis=1)
                    mask_aircraft = np.roll(mask_aircraft, shift=shift_cols, axis=1)

            return np.asarray(xyz_hw3, dtype=np.float32), mask_aircraft, int(H), int(W), ""
    except Exception as e:
        return None, None, 0, 0, f"h5_read_failed:{e}"


def _parse_class_map(raw: str) -> Dict[str, int]:
    s = str(raw or "").strip()
    if not s:
        raise RuntimeError("Empty --class-map.")
    out: Dict[str, int] = {}
    for tok in s.split(","):
        t = str(tok).strip()
        if not t:
            continue
        if ":" not in t:
            raise RuntimeError(f"Invalid class-map token '{t}' (missing ':').")
        k, v = t.split(":", 1)
        key = str(k).strip()
        try:
            cls_id = int(v.strip())
        except Exception as e:
            raise RuntimeError(f"Invalid class id in class-map token '{t}': {e}") from e
        if not key:
            raise RuntimeError(f"Invalid class-map token '{t}' (empty key).")
        if cls_id < 1 or cls_id > 255:
            raise RuntimeError(f"Class id out of range [1,255] in '{t}'.")
        out[key] = int(cls_id)
    if not out:
        raise RuntimeError("No valid entries in --class-map.")
    return out


def _load_edits_json(edits_json_path: str) -> Dict[str, Any]:
    raw = str(edits_json_path or "").strip()
    if not raw:
        return {}
    p = Path(raw).expanduser()
    if not p.exists() or not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def _apply_scene_edits_to_specs(
    specs: Dict[str, Dict[str, Any]],
    *,
    unique_scene: str,
    edits_data: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for k, v in specs.items():
        out[str(k)] = {
            "center_world": np.asarray(v["center_world"], dtype=np.float64).reshape(3).copy(),
            "half": np.asarray(v["half"], dtype=np.float64).reshape(3).copy(),
            "R_world_to_box": np.asarray(v["R_world_to_box"], dtype=np.float64).reshape(3, 3).copy(),
            "source_name": str(v.get("source_name", "")),
        }

    scene_edits = edits_data.get(str(unique_scene), None)
    if not isinstance(scene_edits, dict):
        return out

    for box_key, rec in scene_edits.items():
        if not isinstance(rec, dict):
            continue
        if str(box_key) not in out:
            out[str(box_key)] = {
                "center_world": np.zeros((3,), dtype=np.float64),
                "half": np.ones((3,), dtype=np.float64),
                "R_world_to_box": np.eye(3, dtype=np.float64),
                "source_name": "",
            }
        tgt = out[str(box_key)]
        try:
            if "center_world" in rec:
                tgt["center_world"] = np.asarray(rec["center_world"], dtype=np.float64).reshape(3)
            if "half" in rec:
                h = np.asarray(rec["half"], dtype=np.float64).reshape(3)
                tgt["half"] = np.maximum(h, 1e-6)
            if "R_world_to_box" in rec:
                R = np.asarray(rec["R_world_to_box"], dtype=np.float64).reshape(3, 3)
                tgt["R_world_to_box"] = R
            if "source_name" in rec:
                tgt["source_name"] = str(rec.get("source_name", ""))
        except Exception:
            continue

    return out


def _scene_is_a380(unique_scene: str) -> bool:
    return "a380_800" in str(unique_scene).lower()


def _resolve_det_label_path(
    *,
    det_dataset_root: Path,
    unique_scene: str,
    image_path: Path,
    split_arg: str,
) -> Optional[Path]:
    if not det_dataset_root.exists() or not det_dataset_root.is_dir():
        return None

    splits: List[str] = []
    s = str(split_arg or "auto").strip().lower()
    if s in ("train", "val", "test"):
        splits.append(s)
    else:
        splits.append(_resolve_yolo_split(image_path, "auto"))
        for sp in ("train", "val", "test"):
            if sp not in splits:
                splits.append(sp)

    for sp in splits:
        fp = det_dataset_root / "labels" / sp / f"{unique_scene}.txt"
        if fp.exists() and fp.is_file():
            return fp
    return None


def _load_best_front_bbox_norm(
    label_path: Path,
    *,
    front_class_id: int,
) -> Optional[Tuple[float, float, float, float]]:
    try:
        lines = label_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return None

    best: Optional[Tuple[float, float, float, float]] = None
    best_area = -1.0
    for ln in lines:
        parts = str(ln).strip().split()
        if len(parts) < 5:
            continue
        try:
            cls_id = int(float(parts[0]))
            xc = float(parts[1])
            yc = float(parts[2])
            bw = float(parts[3])
            bh = float(parts[4])
        except Exception:
            continue
        if int(cls_id) != int(front_class_id):
            continue
        if not np.isfinite([xc, yc, bw, bh]).all():
            continue
        if bw <= 0.0 or bh <= 0.0:
            continue
        area = float(bw * bh)
        if area > best_area:
            best_area = area
            best = (xc, yc, bw, bh)
    return best


def _bbox_norm_to_xyxy(
    bbox_norm: Tuple[float, float, float, float],
    *,
    W: int,
    H: int,
) -> Optional[Tuple[int, int, int, int]]:
    if W <= 1 or H <= 1:
        return None
    xc, yc, bw, bh = [float(v) for v in bbox_norm]
    x1 = int(np.floor((xc - 0.5 * bw) * float(W)))
    y1 = int(np.floor((yc - 0.5 * bh) * float(H)))
    x2 = int(np.ceil((xc + 0.5 * bw) * float(W))) - 1
    y2 = int(np.ceil((yc + 0.5 * bh) * float(H))) - 1
    x1 = int(np.clip(x1, 0, W - 1))
    y1 = int(np.clip(y1, 0, H - 1))
    x2 = int(np.clip(x2, 0, W - 1))
    y2 = int(np.clip(y2, 0, H - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _extract_xyz_points_from_bbox(
    xyz_hw3: np.ndarray,
    *,
    bbox_xyxy: Tuple[int, int, int, int],
) -> np.ndarray:
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    roi = np.asarray(xyz_hw3, dtype=np.float64)[y1 : y2 + 1, x1 : x2 + 1, :].reshape(-1, 3)
    if roi.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    finite = np.all(np.isfinite(roi), axis=1)
    return roi[finite]


def _fit_spec_from_points_with_fixed_orientation(
    *,
    points_world: np.ndarray,
    base_spec: Dict[str, Any],
    half_expand: float,
) -> Optional[Dict[str, Any]]:
    pts = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] < 3:
        return None

    c0 = np.asarray(base_spec["center_world"], dtype=np.float64).reshape(3)
    Rwb = np.asarray(base_spec["R_world_to_box"], dtype=np.float64).reshape(3, 3)
    local = (pts - c0.reshape(1, 3)) @ Rwb
    if local.shape[0] >= 40:
        lo = np.percentile(local, 5.0, axis=0)
        hi = np.percentile(local, 95.0, axis=0)
    else:
        lo = np.min(local, axis=0)
        hi = np.max(local, axis=0)

    ctr_local = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    half = np.maximum(half * float(max(1e-6, half_expand)), 1e-3)
    c_new = c0 + (ctr_local @ Rwb.T)
    return {
        "source_name": str(base_spec.get("source_name", "")),
        "center_world": c_new,
        "half": half,
        "R_world_to_box": Rwb.copy(),
    }


def _apply_a380_front_spec_override_from_detection(
    *,
    specs: Dict[str, Dict[str, Any]],
    unique_scene: str,
    image_path: Path,
    xyz_hw3: np.ndarray,
    det_dataset_root: str,
    front_class_id: int,
    split_arg: str,
    half_expand: float,
    min_points: int,
) -> Tuple[Dict[str, Dict[str, Any]], str]:
    out: Dict[str, Dict[str, Any]] = {}
    for k, v in specs.items():
        out[str(k)] = {
            "center_world": np.asarray(v["center_world"], dtype=np.float64).reshape(3).copy(),
            "half": np.asarray(v["half"], dtype=np.float64).reshape(3).copy(),
            "R_world_to_box": np.asarray(v["R_world_to_box"], dtype=np.float64).reshape(3, 3).copy(),
            "source_name": str(v.get("source_name", "")),
        }
    if not _scene_is_a380(unique_scene):
        return out, "not_a380"
    if "front_landing_gear" not in out:
        return out, "front_spec_missing"

    det_root = Path(str(det_dataset_root)).expanduser().resolve()
    lbl_fp = _resolve_det_label_path(
        det_dataset_root=det_root,
        unique_scene=str(unique_scene),
        image_path=Path(image_path),
        split_arg=str(split_arg),
    )
    if lbl_fp is None:
        return out, "det_label_missing"

    H = int(xyz_hw3.shape[0])
    W = int(xyz_hw3.shape[1])
    bbox_norm = _load_best_front_bbox_norm(lbl_fp, front_class_id=int(front_class_id))
    if bbox_norm is None:
        return out, "front_bbox_missing"
    bb = _bbox_norm_to_xyxy(bbox_norm, W=W, H=H)
    if bb is None:
        return out, "front_bbox_invalid"

    pts = _extract_xyz_points_from_bbox(np.asarray(xyz_hw3, dtype=np.float32), bbox_xyxy=bb)
    if pts.shape[0] < int(max(3, min_points)):
        return out, f"front_bbox_points_lt_{int(max(3, min_points))}"

    spec_fit = _fit_spec_from_points_with_fixed_orientation(
        points_world=pts,
        base_spec=out["front_landing_gear"],
        half_expand=float(half_expand),
    )
    if spec_fit is None:
        return out, "front_fit_failed"

    spec_fit["source_name"] = str(out["front_landing_gear"].get("source_name", "front_landing_gear")) + "|det_bbox_a380"
    out["front_landing_gear"] = spec_fit
    return out, f"applied:{lbl_fp}"


def _build_warning_state(args: argparse.Namespace, view_mod: Any) -> Dict[str, Any]:
    prof_csv = str(args.warning_profile_csv or "").strip()
    if not prof_csv:
        prof_csv = str(getattr(view_mod, "WARNING_PROFILE_CSV", "") or "")

    yaml_col = str(args.warning_yaml_column or "").strip()
    if not yaml_col:
        yaml_col = str(getattr(view_mod, "WARNING_YAML_COLUMN", "recommended_yaml"))

    yaml_root = str(args.warning_yaml_root or "").strip()
    if not yaml_root:
        yaml_root = str(getattr(view_mod, "WARNING_YAML_ROOT", "") or "")

    yaml_rel = str(args.warning_yaml_relpath or "").strip()
    if not yaml_rel:
        yaml_rel = str(getattr(view_mod, "WARNING_YAML_RELPATH", "detection_configs/default.yaml"))

    target_level = int(args.warning_target_level)
    if target_level <= 0:
        target_level = int(getattr(view_mod, "WARNING_TARGET_LEVEL", 5))

    center_off = tuple(getattr(view_mod, "WARNING_CENTER_FRAME_OFFSET", (0.0, 0.0, 0.0)))

    return {
        "profile_map": view_mod._load_warning_profile_map(str(prof_csv)),
        "yaml_cache": {},
        "h5_path_cache": {},
        "scene_keypoints_cache": {},
        "warning_yaml_column": str(yaml_col),
        "warning_yaml_root": str(yaml_root),
        "warning_yaml_relpath": str(yaml_rel),
        "warning_target_level": int(target_level),
        "warning_center_frame_offset": center_off,
        "warning_h5_root": str(args.source),
        "use_scene_h5_transform": True,
    }


def _make_mask_from_specs(
    xyz_hw3: np.ndarray,
    specs: Dict[str, Dict[str, Any]],
    class_map: Dict[str, int],
) -> np.ndarray:
    H, W = int(xyz_hw3.shape[0]), int(xyz_hw3.shape[1])
    n = H * W
    pts = np.asarray(xyz_hw3, dtype=np.float64).reshape(-1, 3)
    finite = np.all(np.isfinite(pts), axis=1)

    best_score = np.full((n,), np.inf, dtype=np.float64)
    class_flat = np.zeros((n,), dtype=np.uint8)
    for box_key, cls_id in class_map.items():
        spec = specs.get(str(box_key), None)
        if spec is None:
            continue

        c = np.asarray(spec["center_world"], dtype=np.float64).reshape(1, 3)
        Rwb = np.asarray(spec["R_world_to_box"], dtype=np.float64).reshape(3, 3)
        h = np.asarray(spec["half"], dtype=np.float64).reshape(1, 3)
        denom = np.maximum(h, 1e-9)

        local = np.full((n, 3), np.nan, dtype=np.float64)
        local[finite] = (pts[finite] - c) @ Rwb
        inside = np.zeros((n,), dtype=bool)
        inside[finite] = np.all(np.abs(local[finite]) <= (h + 1e-6), axis=1)

        # Overlap tie-breaker: prefer lower normalized box-local infinity norm.
        score = np.full((n,), np.inf, dtype=np.float64)
        score[inside] = np.max(np.abs(local[inside]) / denom, axis=1)
        better = score < best_score
        class_flat[better] = np.uint8(cls_id)
        best_score[better] = score[better]

    return class_flat.reshape(H, W)


def _apply_aircraft_mask(
    mask_hw: np.ndarray,
    aircraft_mask_hw: Optional[np.ndarray],
    *,
    aircraft_class_id: int,
    fill_background_only: bool = True,
) -> np.ndarray:
    out = np.asarray(mask_hw, dtype=np.uint8).copy()
    if aircraft_mask_hw is None:
        return out
    m = np.asarray(aircraft_mask_hw, dtype=bool)
    if m.ndim != 2 or out.shape[:2] != m.shape[:2]:
        return out
    cls = int(np.clip(int(aircraft_class_id), 1, 255))
    if bool(fill_background_only):
        write = m & (out == 0)
        out[write] = np.uint8(cls)
    else:
        out[m] = np.uint8(cls)
    return out


def _make_overlay(
    image_bgr: np.ndarray,
    mask_hw: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    out = np.asarray(image_bgr, dtype=np.uint8).copy()
    overlay = out.copy()
    palette = {
        1: (255, 0, 255),    # front
        2: (0, 255, 0),      # left (green)
        3: (0, 200, 255),    # right
        4: (255, 0, 0),      # aircraft (blue in BGR)
        5: (255, 128, 128),
        6: (128, 255, 128),
        7: (128, 128, 255),
    }
    cls_vals = np.unique(mask_hw)
    for v in cls_vals.tolist():
        cls_id = int(v)
        if cls_id <= 0:
            continue
        col = palette.get(cls_id, (180, 180, 180))
        m = mask_hw == cls_id
        overlay[m] = col
    return cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0.0)


def _resolve_yolo_split(image_path: Path, split_arg: str) -> str:
    s = str(split_arg or "auto").strip().lower()
    if s in ("train", "val", "test"):
        return s
    parts = [str(p).strip().lower() for p in image_path.parts]
    for p in reversed(parts):
        if p in ("train", "val", "test"):
            return p
    return "val"


def _build_yolo_class_layout(
    class_map: Dict[str, int],
    *,
    include_aircraft_mask: bool,
    aircraft_class_id: int,
) -> Tuple[Dict[int, int], Dict[int, str]]:
    class_id_to_yolo: Dict[int, int] = {}
    yolo_idx_to_name: Dict[int, str] = {}

    ordered_ids: List[int] = []
    ordered_names: List[str] = []
    for key, cls_id in class_map.items():
        cid = int(cls_id)
        if cid in ordered_ids:
            continue
        ordered_ids.append(cid)
        ordered_names.append(str(key))

    if include_aircraft_mask:
        ac = int(aircraft_class_id)
        if ac not in ordered_ids:
            ordered_ids.append(ac)
            ordered_names.append("aircraft")

    for i, cid in enumerate(ordered_ids):
        class_id_to_yolo[int(cid)] = int(i)
        yolo_idx_to_name[int(i)] = str(ordered_names[i])
    return class_id_to_yolo, yolo_idx_to_name


def _mask_to_yolo_seg_lines(
    mask_hw: np.ndarray,
    *,
    class_id_to_yolo: Dict[int, int],
    min_contour_area: float,
    contour_approx_eps: float,
) -> Tuple[List[str], Dict[int, int]]:
    H, W = int(mask_hw.shape[0]), int(mask_hw.shape[1])
    if H <= 0 or W <= 0:
        return [], {}

    lines: List[str] = []
    inst_count_by_class: Dict[int, int] = {}
    for class_id, yolo_idx in sorted(class_id_to_yolo.items(), key=lambda kv: kv[1]):
        bin_mask = (np.asarray(mask_hw, dtype=np.uint8) == np.uint8(class_id)).astype(np.uint8)
        if int(np.count_nonzero(bin_mask)) <= 0:
            continue

        contours, _hier = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if c is None or len(c) < 3:
                continue
            area = float(abs(cv2.contourArea(c)))
            if area < float(min_contour_area):
                continue

            poly = c
            if float(contour_approx_eps) > 0.0:
                poly = cv2.approxPolyDP(c, epsilon=float(contour_approx_eps), closed=True)
            if poly is None:
                continue
            pts = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
            if int(pts.shape[0]) < 3:
                continue

            vals: List[str] = []
            for x, y in pts:
                xn = float(np.clip(float(x) / float(max(1, W)), 0.0, 1.0))
                yn = float(np.clip(float(y) / float(max(1, H)), 0.0, 1.0))
                vals.append(f"{xn:.6f}")
                vals.append(f"{yn:.6f}")
            if len(vals) < 6:
                continue

            lines.append(f"{int(yolo_idx)} " + " ".join(vals))
            inst_count_by_class[int(yolo_idx)] = int(inst_count_by_class.get(int(yolo_idx), 0) + 1)

    return lines, inst_count_by_class


def _write_yolo_dataset_yaml(yolo_root: Path, yaml_name: str, yolo_idx_to_name: Dict[int, str]) -> Path:
    fp = yolo_root / str(yaml_name)
    lines = [
        f"path: {yolo_root}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        "names:",
    ]
    for idx in sorted(yolo_idx_to_name.keys()):
        lines.append(f"  {int(idx)}: {str(yolo_idx_to_name[idx])}")
    fp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return fp


def main() -> None:
    args = _parse_args()

    try:
        import test_yolo_pose_from_h5_weights_to_pcd as pose_pcd
        import view_pcd_dir as pcd_view
    except Exception as e:
        raise RuntimeError("Failed importing project modules. Activate your project venv.") from e

    image_path_raw = str(args.image_path or "").strip()
    image_dir_raw = str(args.image_dir or "").strip()
    if image_path_raw and image_dir_raw:
        # Common case: user explicitly sets --image-dir but keeps default image-path.
        if str(image_path_raw) == str(DEFAULT_IMAGE_PATH):
            image_path_raw = ""
        else:
            raise RuntimeError("Use only one of --image-path or --image-dir.")

    class_map = _parse_class_map(str(args.class_map))
    include_aircraft_mask = bool(int(args.include_aircraft_mask))
    aircraft_class_id = int(np.clip(int(args.aircraft_class_id), 1, 255))
    export_yolo_seg = bool(int(args.export_yolo_seg))
    edits_data = _load_edits_json(str(args.edits_json or ""))
    out_root = Path(str(args.out)).expanduser().resolve()
    mask_root = out_root / "masks"
    vis_root = out_root / "vis"
    mask_root.mkdir(parents=True, exist_ok=True)
    if int(args.save_overlay) == 1:
        vis_root.mkdir(parents=True, exist_ok=True)

    yolo_root: Optional[Path] = None
    yolo_class_id_to_idx: Dict[int, int] = {}
    yolo_idx_to_name: Dict[int, str] = {}
    if export_yolo_seg:
        yolo_out_raw = str(args.yolo_dataset_dir or "").strip()
        yolo_root = (
            Path(yolo_out_raw).expanduser().resolve()
            if yolo_out_raw
            else (out_root / "yolo_seg_dataset")
        )
        for rel in (
            "images/train",
            "images/val",
            "images/test",
            "labels/train",
            "labels/val",
            "labels/test",
        ):
            (yolo_root / rel).mkdir(parents=True, exist_ok=True)
        yolo_class_id_to_idx, yolo_idx_to_name = _build_yolo_class_layout(
            class_map=class_map,
            include_aircraft_mask=include_aircraft_mask,
            aircraft_class_id=aircraft_class_id,
        )

    # Resolve input images.
    image_paths: List[Path] = []
    if image_dir_raw:
        image_paths = _collect_images_from_dir(Path(image_dir_raw).expanduser().resolve())
    elif image_path_raw:
        p = Path(image_path_raw).expanduser().resolve()
        if p.exists() and p.is_dir():
            image_paths = _collect_images_from_dir(p)
        elif p.exists() and p.is_file():
            image_paths = [p]
        else:
            raise RuntimeError(f"Input path not found: {p}")
    else:
        raise RuntimeError("Please provide --image-path or --image-dir.")

    max_images = int(args.max_images)
    if max_images > 0:
        image_paths = image_paths[:max_images]
    if not image_paths:
        raise RuntimeError("No images selected.")

    print(f"[input] images={len(image_paths)} source_h5_root={args.source}")
    h5_by_stem = _build_h5_index(str(args.source))
    if not h5_by_stem:
        raise RuntimeError(f"No H5 files found under: {args.source}")

    warning_state = _build_warning_state(args, pcd_view)
    print(
        f"[warning] yaml_column={warning_state['warning_yaml_column']} "
        f"yaml_root={warning_state['warning_yaml_root']} "
        f"yaml_relpath={warning_state['warning_yaml_relpath']} "
        f"target_level={warning_state['warning_target_level']}"
    )
    print(f"[class-map] {class_map}")
    if str(args.edits_json or "").strip():
        print(f"[edits] json={Path(str(args.edits_json)).expanduser()} scenes={len(edits_data)}")
    if include_aircraft_mask:
        print(f"[aircraft-mask] enabled class_id={aircraft_class_id} mode=fill_background_only")
    else:
        print("[aircraft-mask] disabled")
    if export_yolo_seg:
        print(f"[yolo-seg] export=1 root={yolo_root}")
        print(f"[yolo-seg] classes={yolo_idx_to_name}")
    else:
        print("[yolo-seg] export=0")
    use_a380_det_override = bool(int(getattr(args, "a380_front_from_det_bbox", 1)))
    print(
        "[a380-front-override] "
        f"enabled={int(use_a380_det_override)} "
        f"det_root={Path(str(getattr(args, 'a380_det_dataset_root', DEFAULT_A380_DET_DATASET_ROOT))).expanduser()} "
        f"class_id={int(getattr(args, 'a380_det_front_class_id', 3))} "
        f"split={str(getattr(args, 'a380_det_split', 'auto'))}"
    )

    summary_rows: List[Dict[str, Any]] = []
    saved = 0
    skipped_bad_stem = 0
    skipped_no_h5 = 0
    skipped_no_scene_xyz = 0
    skipped_no_warning_specs = 0

    for i, ip in enumerate(image_paths, 1):
        bgr = cv2.imread(str(ip), cv2.IMREAD_COLOR)
        if bgr is None:
            skipped_no_scene_xyz += 1
            print(f"[{i}/{len(image_paths)}] [skip] unreadable image: {ip.name}")
            continue

        try:
            h5_stem, scene_name = _parse_unique_scene_stem(ip.stem)
        except Exception as e:
            skipped_bad_stem += 1
            print(f"[{i}/{len(image_paths)}] [skip] bad scene stem: {ip.name} ({e})")
            continue

        matches = h5_by_stem.get(str(h5_stem), [])
        if not matches:
            skipped_no_h5 += 1
            print(f"[{i}/{len(image_paths)}] [skip] no H5 for stem={h5_stem}")
            continue
        h5_path = matches[0]
        unique_scene = f"{h5_stem}__{scene_name}"

        xyz_hw3, mask_air, H, W, xyz_reason = _load_scene_xyz(
            h5_path=h5_path, scene_name=str(scene_name), pose_mod=pose_pcd
        )
        if xyz_hw3 is None:
            skipped_no_scene_xyz += 1
            print(f"[{i}/{len(image_paths)}] [skip] xyz missing: {unique_scene} ({xyz_reason})")
            continue

        specs, _checks, warn_reason = pcd_view._build_warning_specs_and_checks(
            unique_scene=unique_scene,
            kp_named=[],
            warning_state=warning_state,
            warning_box_scale=float(args.warning_box_scale),
        )
        if not specs:
            skipped_no_warning_specs += 1
            print(f"[{i}/{len(image_paths)}] [skip] warning specs missing: {unique_scene} ({warn_reason})")
            continue
        a380_front_override_status = "disabled"
        if use_a380_det_override:
            specs, a380_front_override_status = _apply_a380_front_spec_override_from_detection(
                specs=specs,
                unique_scene=unique_scene,
                image_path=ip,
                xyz_hw3=np.asarray(xyz_hw3, dtype=np.float32),
                det_dataset_root=str(getattr(args, "a380_det_dataset_root", DEFAULT_A380_DET_DATASET_ROOT)),
                front_class_id=int(getattr(args, "a380_det_front_class_id", 3)),
                split_arg=str(getattr(args, "a380_det_split", "auto")),
                half_expand=float(getattr(args, "a380_det_half_expand", 1.0)),
                min_points=int(getattr(args, "a380_det_min_points", 30)),
            )
        specs = _apply_scene_edits_to_specs(specs=specs, unique_scene=unique_scene, edits_data=edits_data)

        mask_hw = _make_mask_from_specs(
            xyz_hw3=np.asarray(xyz_hw3, dtype=np.float32),
            specs=specs,
            class_map=class_map,
        )
        mask_hw = _apply_aircraft_mask(
            mask_hw=mask_hw,
            aircraft_mask_hw=mask_air if include_aircraft_mask else None,
            aircraft_class_id=int(aircraft_class_id),
            fill_background_only=True,
        )

        mask_fp = mask_root / f"{unique_scene}.png"
        cv2.imwrite(str(mask_fp), np.asarray(mask_hw, dtype=np.uint8))

        src_img = bgr
        if int(src_img.shape[0]) != int(H) or int(src_img.shape[1]) != int(W):
            src_img = cv2.resize(src_img, (int(W), int(H)), interpolation=cv2.INTER_LINEAR)

        if int(args.save_overlay) == 1:
            vis = _make_overlay(src_img, mask_hw, alpha=float(args.overlay_alpha))
            vis_fp = vis_root / f"{unique_scene}.png"
            cv2.imwrite(str(vis_fp), vis)

        yolo_label_fp = ""
        yolo_image_fp = ""
        yolo_split = ""
        yolo_instances = 0
        yolo_inst_counts: Dict[int, int] = {}
        if export_yolo_seg and yolo_root is not None:
            yolo_split = _resolve_yolo_split(ip, str(args.yolo_split))
            yolo_img_dir = yolo_root / "images" / yolo_split
            yolo_lbl_dir = yolo_root / "labels" / yolo_split
            yolo_img_dir.mkdir(parents=True, exist_ok=True)
            yolo_lbl_dir.mkdir(parents=True, exist_ok=True)
            yolo_img = yolo_img_dir / f"{unique_scene}.png"
            yolo_lbl = yolo_lbl_dir / f"{unique_scene}.txt"
            cv2.imwrite(str(yolo_img), src_img)
            yolo_lines, yolo_inst_counts = _mask_to_yolo_seg_lines(
                mask_hw=mask_hw,
                class_id_to_yolo=yolo_class_id_to_idx,
                min_contour_area=float(args.min_contour_area),
                contour_approx_eps=float(args.contour_approx_eps),
            )
            yolo_lbl.write_text(("\n".join(yolo_lines) + "\n") if yolo_lines else "", encoding="utf-8")
            yolo_label_fp = str(yolo_lbl)
            yolo_image_fp = str(yolo_img)
            yolo_instances = int(len(yolo_lines))

        row: Dict[str, Any] = {
            "unique_scene": unique_scene,
            "h5_file": str(h5_path.name),
            "scene_name": str(scene_name),
            "mask_path": str(mask_fp),
            "pixels_labeled": int(np.count_nonzero(mask_hw > 0)),
            "warning_reason": str(warn_reason or ""),
            "a380_front_override": str(a380_front_override_status),
        }
        if export_yolo_seg:
            row["yolo_split"] = str(yolo_split)
            row["yolo_image_path"] = str(yolo_image_fp)
            row["yolo_label_path"] = str(yolo_label_fp)
            row["yolo_instances"] = int(yolo_instances)
        for k, cls_id in class_map.items():
            pix_cls = int(np.count_nonzero(mask_hw == int(cls_id)))
            row[f"class_{cls_id}_{k}_pixels"] = pix_cls
            row[f"spec_{k}_exists"] = int(k in specs)
            row[f"count_{k}"] = pix_cls
        if include_aircraft_mask:
            row[f"class_{aircraft_class_id}_aircraft_pixels"] = int(
                np.count_nonzero(mask_hw == int(aircraft_class_id))
            )
            row["aircraft_mask_available"] = int(mask_air is not None)
        if export_yolo_seg:
            for idx, name in sorted(yolo_idx_to_name.items(), key=lambda kv: kv[0]):
                row[f"yolo_class_{idx}_{name}_instances"] = int(yolo_inst_counts.get(int(idx), 0))
        summary_rows.append(row)
        saved += 1
        print(f"[{i}/{len(image_paths)}] [ok] {unique_scene} labeled_px={row['pixels_labeled']}")

    summary_csv = out_root / "warning_box_segmentation_summary.csv"
    headers = [
        "unique_scene",
        "h5_file",
        "scene_name",
        "mask_path",
        "pixels_labeled",
        "warning_reason",
        "a380_front_override",
    ]
    if export_yolo_seg:
        headers.extend(
            [
                "yolo_split",
                "yolo_image_path",
                "yolo_label_path",
                "yolo_instances",
            ]
        )
    for k, cls_id in class_map.items():
        headers.extend(
            [
                f"class_{cls_id}_{k}_pixels",
                f"spec_{k}_exists",
                f"count_{k}",
            ]
        )
    if include_aircraft_mask:
        headers.extend(
            [
                f"class_{aircraft_class_id}_aircraft_pixels",
                "aircraft_mask_available",
            ]
        )
    if export_yolo_seg:
        for idx, name in sorted(yolo_idx_to_name.items(), key=lambda kv: kv[0]):
            headers.append(f"yolo_class_{idx}_{name}_instances")

    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    print(f"[out] masks={mask_root}")
    if int(args.save_overlay) == 1:
        print(f"[out] overlays={vis_root}")
    if export_yolo_seg and yolo_root is not None:
        yolo_yaml = _write_yolo_dataset_yaml(
            yolo_root=yolo_root,
            yaml_name=str(args.yolo_yaml_name),
            yolo_idx_to_name=yolo_idx_to_name,
        )
        print(f"[out] yolo_dataset={yolo_root}")
        print(f"[out] yolo_yaml={yolo_yaml}")
    print(f"[out] summary={summary_csv}")
    print(
        "[summary] "
        f"saved={saved} "
        f"skipped_bad_stem={skipped_bad_stem} "
        f"skipped_no_h5={skipped_no_h5} "
        f"skipped_no_scene_xyz={skipped_no_scene_xyz} "
        f"skipped_no_warning_specs={skipped_no_warning_specs}"
    )


if __name__ == "__main__":
    main()
