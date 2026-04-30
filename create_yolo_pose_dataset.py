#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create a unified YOLO-pose dataset from many aircraft HDF5 LiDAR range-image files.

This version uses a **BAG-LEVEL SPLIT** (by H5 file), so TEST contains bags (H5 files)
that were not seen in TRAIN/VAL.

Because each H5 can have a different number of scenes, the split is **balanced by scene count**
(whole files are assigned to train/val/test to approximately match the desired scene ratios).
"""

import json
import argparse
import random
import csv
import ast
import math
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from collections import defaultdict

import numpy as np
import h5py
import imageio.v2 as imageio
import cv2

from config_dataset import (
    SOURCE,
    OUT_DIR,
    SPLIT,
    RANDOM_SEED,
    MAX_H5_FILES,
    REUSE_LABELS_FROM_DIR,
    REUSE_LABELS_STRICT,
    DRAW_ON_OVERLAY,
    APPLY_MEDIAN_FILTER,
    MEDIAN_KSIZE,
    IMAGE_RENDER_MODE,
    IMAGE_CHANNEL_FIELDS,
    IMAGE_COLORMAP,
    BLUE_CHANNEL_MODE,
    BLUE_CHANNEL_GAMMA,
    FAR_BRIGHT_BOOST_ENABLE,
    FAR_BRIGHT_BOOST_FIELDS,
    FAR_BRIGHT_BOOST_STRENGTH,
    INTENSITY_BOOST_ENABLE,
    INTENSITY_BOOST_GAIN,
    INTENSITY_ROW_CORRECTION_ENABLE,
    INTENSITY_ROW_CORRECTION_STRENGTH,
    INTENSITY_ROW_CORRECTION_SIGMA_ROWS,
    INTENSITY_ROW_CORRECTION_MAX_SHIFT,
    EXPORT_SINGLE_CHANNEL_IMAGE,
    SINGLE_CHANNEL_FIELD,
    GROUND_SEPARATION_ENABLE,
    GROUND_ATTENUATION_FACTOR,
    USE_TF_MATRIX,
    APPLY_Z_FLIP,
    SYN_KP_NAME,
    REMOVE_KP_SET,
    FRONT_RIGHT_ALIASES,
    FRONT_LEFT_ALIASES,
    USE_NOSE_GEAR_CENTER_FOR_SYNTHETIC_FRONT_MID,
    NOSE_GEAR_ALIASES,
    ADD_ENGINE_WARNING_BOX_KEYPOINTS,
    ENGINE_LEFT_KP_NAME,
    ENGINE_RIGHT_KP_NAME,
    ENGINE_LEFT_BOX_ALIASES,
    ENGINE_RIGHT_BOX_ALIASES,
    USE_WARNING_BOX_KEYPOINTS,
    WARNING_PROFILE_CSV,
    WARNING_YAML_COLUMN,
    WARNING_YAML_ROOT,
    WARNING_YAML_RELPATH,
    WARNING_CENTER_KEYPOINT_NAME,
    WARNING_TARGET_LEVEL,
    WARNING_CENTER_FRAME_OFFSET,
    WARNING_ENGINE_Z_OFFSET,
    WARNING_LANDING_GEAR_Z_OFFSET,
    WARNING_WING_Z_OFFSET,
    WARNING_REAR_WING_Z_OFFSET,
    WARNING_DERIVE_FRONT_GEAR_FROM_WHEELS,
    WARNING_FRONT_GEAR_NAME_FILTERS,
    WARNING_ENGINE_LEFT_NAME_FILTERS,
    WARNING_ENGINE_RIGHT_NAME_FILTERS,
    ENGINE_BOX_SNAP_ENABLED,
    ENGINE_BOX_SNAP_MIN_POINTS,
    ENGINE_BOX_SNAP_EXPAND_FACTOR,
    ENGINE_BOX_SNAP_MAX_DRIFT_M,
    ENGINE_PIXEL_SNAP_ENABLED,
    ENGINE_PIXEL_SNAP_RADIUS,
    ENGINE_PIXEL_SNAP_FALLBACK_TO_NEAREST,
    ENGINE_PIXEL_SNAP_FALLBACK_MAX_DIST,
    ENGINE_PIXEL_ROW_BIAS,
    MAKE_VIZ,
    DEBUG_POINTCLOUD_KEYPOINTS,
    DEBUG_POINTCLOUD_MAX_SCENES,
    DEBUG_POINTCLOUD_SAMPLE_EVERY_N,
    DEBUG_POINTCLOUD_MAX_POINTS,
    DEBUG_POINTCLOUD_LIVE_VIEWER,
    DRAW_ENGINE_VIS_BBOX,
    ENGINE_VIS_BBOX_HALF_W,
    ENGINE_VIS_BBOX_HALF_H,
    DRAW_NOSE_GEAR_VIS_BBOX,
    NOSE_VIS_BBOX_HALF_W,
    NOSE_VIS_BBOX_HALF_H,
    KPT_BBOX_MARGIN_PX,
    ROLL_WIDE_BBOX,
    ROLL_WIDE_BBOX_FRAC,
    ROLL_WIDE_BBOX_COLS,
    RAY_VISIBILITY_CHECK,
    RAY_TOL,
    RAY_PATCH_RADIUS,
    RAY_REQUIRE_LOCAL_HIT_KEYPOINTS,
    RAY_VISIBILITY_EXEMPT_KEYPOINTS,
    MID_BASE_RADIUS,
    MID_EXPAND_RADIUS,
    MID_Z_BAND,
    MID_MIN_POINTS,
)

from io_helpers import list_h5_paths, open_h5_any, write_bag_split_columns_csv
from projection_helpers import (
    build_rgb_from_cols,
    build_gray_from_cols,
    bbox_from_mask,
    xyxy_to_xywhn,
    get_min_depth,
    adjust_midpoint_to_cluster,
    angles_from_xyz,
    row_from_elevation,
    col_from_azimuth_global,
    apply_transform,
    rc_to_xy_norm,
    find_alias,
    _fill_nans,
)

RAY_VISIBILITY_EXEMPT_SET = {
    str(name).strip()
    for name in RAY_VISIBILITY_EXEMPT_KEYPOINTS
    if str(name).strip()
}
RAY_REQUIRE_LOCAL_HIT_SET = {
    str(name).strip()
    for name in RAY_REQUIRE_LOCAL_HIT_KEYPOINTS
    if str(name).strip()
}


# =========================
# Helper: smart azimuth seam placement
# =========================
def find_best_azimuth_roll(mask2d: np.ndarray) -> int:
    """
    Given a boolean is_aircraft mask (H, W), find a horizontal roll (shift)
    that places the panorama seam (column 0) in the largest empty-azimuth gap.

    Returns:
        shift (int): number of columns to np.roll(..., shift=shift, axis=1)

    If we cannot find any empty columns, or rolling does not help, returns 0.
    """
    H, W = mask2d.shape
    col_has_aircraft = mask2d.any(axis=0)  # (W,)
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


def _wrap_aware_bbox(mask2d: np.ndarray) -> tuple[tuple[int, int, int, int] | None, int]:
    """
    Compute wrap-aware bbox by testing a seam shift on circular azimuth.

    Returns:
      (bbox, shift)
      - bbox is in shifted image coordinates.
      - shift is columns for np.roll(..., axis=1). 0 means no shift applied.
    """
    bb = bbox_from_mask(mask2d)
    if bb is None:
        return None, 0

    W = mask2d.shape[1]
    if W <= 1:
        return bb, 0

    x1, y1, x2, y2 = bb
    w0 = x2 - x1 + 1

    shift = find_best_azimuth_roll(mask2d)
    if shift == 0:
        return bb, 0

    rolled_mask = np.roll(mask2d, shift=shift, axis=1)
    bb2 = bbox_from_mask(rolled_mask)
    if bb2 is None:
        return bb, 0

    w2 = bb2[2] - bb2[0] + 1
    if w2 < w0:
        return bb2, shift
    return bb, 0


def _resolve_reuse_root(path_raw: str | None) -> Path | None:
    s = str(path_raw or "").strip()
    if not s:
        return None
    return Path(s).expanduser().resolve()


def _build_reuse_label_index(root: Path) -> dict[str, list[Path]]:
    """
    Build scene-stem -> candidate label files index.
    Supports:
      - dataset root containing labels/train|val|test
      - labels directory directly
      - any directory containing *.txt label files
    """
    start = root / "labels" if (root / "labels").is_dir() else root
    out: dict[str, list[Path]] = defaultdict(list)
    for p in start.rglob("*.txt"):
        if p.is_file():
            out[p.stem].append(p.resolve())
    return out


def _pick_reuse_label_path(
    reuse_root: Path,
    reuse_index: dict[str, list[Path]],
    split_name: str,
    unique_scene: str,
) -> Path | None:
    # Fast direct probes first.
    direct = []
    if (reuse_root / "labels").is_dir():
        direct.extend(
            [
                reuse_root / "labels" / split_name / f"{unique_scene}.txt",
                reuse_root / "labels" / f"{unique_scene}.txt",
            ]
        )
    direct.extend(
        [
            reuse_root / split_name / f"{unique_scene}.txt",
            reuse_root / f"{unique_scene}.txt",
        ]
    )
    for p in direct:
        if p.is_file():
            return p.resolve()

    cands = reuse_index.get(unique_scene, [])
    if not cands:
        return None
    if len(cands) == 1:
        return cands[0]

    # Deterministic preference: labels/..., then matching split folder.
    def _score(p: Path) -> tuple[int, int, str]:
        parts = [x.lower() for x in p.parts]
        score = 0
        if "labels" in parts:
            score += 10
        if split_name.lower() in parts:
            score += 5
        # Tie-break by shorter path string.
        return (score, -len(str(p)), str(p))

    return sorted(cands, key=_score, reverse=True)[0]


def _load_yaml_keypoint_names(yaml_path: Path) -> list[str]:
    if not yaml_path.exists() or not yaml_path.is_file():
        return []
    lines = yaml_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    names: list[str] = []
    in_block = False
    for ln in lines:
        s = ln.strip()
        if s == "keypoints:":
            in_block = True
            continue
        if in_block:
            if s.startswith("- "):
                nm = s[2:].strip()
                if nm:
                    names.append(nm)
            elif s and not s.startswith("#"):
                break
    return names


def _prepare_reuse_pose_label_text(
    raw_text: str,
    target_kp_names: list[str],
    source_kp_names: list[str] | None = None,
) -> tuple[str | None, str, bool]:
    lines = [ln.strip() for ln in str(raw_text).splitlines() if ln.strip()]
    if not lines:
        return None, "empty label file", False

    nk_target = len(target_kp_names)
    src_names = [str(x).strip() for x in (source_kp_names or []) if str(x).strip()]
    src_idx = {nm: i for i, nm in enumerate(src_names)} if src_names else {}
    remapped_any = False
    out_lines: list[str] = []
    for i, ln in enumerate(lines, 1):
        toks = ln.split()
        if len(toks) < 5:
            return None, f"line {i}: expected at least 5 tokens", False
        rem = toks[5:]
        if len(rem) % 3 != 0:
            return None, f"line {i}: keypoint payload not divisible by 3", False
        n_kps = len(rem) // 3
        if n_kps == nk_target:
            out_lines.append(ln)
            continue

        if src_names and n_kps == len(src_names):
            remapped: list[str] = []
            for kp in target_kp_names:
                j = src_idx.get(kp)
                if j is None:
                    remapped.extend(["0.000000", "0.000000", "0"])
                else:
                    remapped.extend(rem[3 * j : 3 * j + 3])
            out_lines.append(" ".join(toks[:5] + remapped))
            remapped_any = True
            continue

        return (
            None,
            f"line {i}: keypoint count {n_kps} not compatible with target {nk_target}",
            False,
        )

    return ("\n".join(out_lines) + "\n"), "", remapped_any


def _norm_aircraft_key(v: str | None) -> str:
    s = str(v or "").strip().lower()
    s = re.sub(r"[\s\-]+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _extract_aircraft_token_from_bag(bag_name: str) -> str:
    m = re.match(r"^movement_(.+?)(?:__|$)", str(bag_name), flags=re.IGNORECASE)
    if not m:
        return str(bag_name)
    return m.group(1).strip("_")


def _infer_profile_from_h5_path(h5_path: str) -> str:
    s = str(h5_path)
    if s.startswith("gs://"):
        parts = s[len("gs://"):].split("/")
        return parts[-2] if len(parts) >= 2 else ""
    return Path(s).parent.name


def _slugify_aircraft_name(name: str) -> str:
    s = str(name).strip().lower()
    s = re.sub(r"[\s\-]+", "_", s)
    s = re.sub(r"[^a-z0-9_+]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _iter_warning_yaml_slug_candidates(bag_name: str, profile: str | None) -> list[str]:
    seeds: list[str] = []
    if profile:
        seeds.append(profile)
    seeds.append(_extract_aircraft_token_from_bag(bag_name))

    out: list[str] = []
    seen: set[str] = set()

    def _add(v: str | None) -> None:
        if not v:
            return
        s = _slugify_aircraft_name(v)
        if not s or s in seen:
            return
        seen.add(s)
        out.append(s)

    for seed in seeds:
        base = str(seed).strip()
        if not base:
            continue

        _add(base)
        _add(base.replace("_", " "))
        _add(base.replace("_", "-"))
        _add(re.sub(r"([A-Za-z])(\d)", r"\1_\2", base))
        _add(re.sub(r"(\d)([A-Za-z])", r"\1_\2", base))

        if re.match(r"^[bB]\d", base):
            b = base[1:]
            _add(b)
            _add(b.replace("_", " "))
            _add(b.replace("_", "-"))

    expanded: list[str] = []
    seen2: set[str] = set()
    for s in out:
        variants = [s, s.replace("max_", "max")]
        if s.endswith("_split"):
            variants.append(s[: -len("_split")])
        for v in variants:
            if v and v not in seen2:
                seen2.add(v)
                expanded.append(v)
    return expanded


def _resolve_warning_yaml_for_bag(
    bag_name: str,
    profile: str | None,
) -> Path | None:
    root = Path(WARNING_YAML_ROOT).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        return None

    rel = str(WARNING_YAML_RELPATH or "detection_configs/default.yaml").strip("/")
    for slug in _iter_warning_yaml_slug_candidates(bag_name=bag_name, profile=profile):
        p = root / slug / rel
        if p.exists():
            return p
    return None


def _load_warning_profile_map(csv_path: str) -> dict[str, dict[str, str]]:
    p = Path(csv_path).expanduser()
    if not p.exists() or not p.is_file():
        return {}
    out: dict[str, dict[str, str]] = {}
    with p.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not isinstance(row, dict):
                continue
            keys = {
                _norm_aircraft_key(row.get("profile_name", "")),
                _norm_aircraft_key(row.get("aircraft", "")),
            }
            keys = {k for k in keys if k}
            for k in keys:
                if k not in out:
                    out[k] = row
    return out


def _lookup_warning_profile_entry(
    mapping: dict[str, dict[str, str]],
    profile: str | None,
    bag_name: str,
) -> dict[str, str] | None:
    if not mapping:
        return None
    cands = [profile, _extract_aircraft_token_from_bag(bag_name)]
    for c in cands:
        k = _norm_aircraft_key(c)
        if k and k in mapping:
            return mapping[k]
    return None


def _parse_csv_box_name_filters(row: dict[str, str] | None) -> list[str]:
    if not row:
        return []
    raw_py = str(row.get("box_name_filters_py", "") or "").strip()
    if raw_py:
        try:
            obj = ast.literal_eval(raw_py)
            if isinstance(obj, list):
                return [str(x).strip() for x in obj if str(x).strip()]
        except Exception:
            pass
    raw = str(row.get("box_name_filters", "") or "").strip()
    if raw:
        return [x.strip() for x in raw.split("|") if x.strip()]
    return []


def _load_yaml_crop_boxes(yaml_path: Path, cache: dict[Path, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    yp = yaml_path.expanduser().resolve()
    if yp in cache:
        return cache[yp]
    try:
        import yaml
    except Exception:
        cache[yp] = []
        return cache[yp]

    if not yp.exists():
        cache[yp] = []
        return cache[yp]

    try:
        data = yaml.safe_load(yp.read_text(encoding="utf-8"))
        boxes = []
        if isinstance(data, dict) and isinstance(data.get("crop_boxes"), dict):
            for name, box in data["crop_boxes"].items():
                if not isinstance(box, dict):
                    continue
                try:
                    boxes.append({
                        "name": str(name),
                        "warning_level": int(box.get("warning_level", -1)),
                        "x": float(box["x"]),
                        "y": float(box["y"]),
                        "z": float(box["z"]),
                        "rx": float(box.get("rx", 0.0)),
                        "ry": float(box.get("ry", 0.0)),
                        "rz": float(box.get("rz", 0.0)),
                        "sx": float(box.get("sx", 0.0)),
                        "sy": float(box.get("sy", 0.0)),
                        "sz": float(box.get("sz", 0.0)),
                    })
                except Exception:
                    continue
        cache[yp] = boxes
        return boxes
    except Exception:
        cache[yp] = []
        return cache[yp]


def _normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        raise ValueError("Cannot normalize near-zero vector")
    return v / n


def _estimate_target_from_center_rotation_from_arrays(names: list[str], xyz: np.ndarray) -> np.ndarray | None:
    req = [
        "left_wing_tip",
        "right_wing_tip",
        "plane_front_left_wheel_link",
        "plane_front_right_wheel_link",
        "plane_rear_left_wheel_link",
        "plane_rear_right_wheel_link",
    ]
    name_to_idx = {str(n): i for i, n in enumerate(names)}
    if any(k not in name_to_idx for k in req):
        return None
    left_tip = xyz[name_to_idx["left_wing_tip"]]
    right_tip = xyz[name_to_idx["right_wing_tip"]]
    front_left = xyz[name_to_idx["plane_front_left_wheel_link"]]
    front_right = xyz[name_to_idx["plane_front_right_wheel_link"]]
    rear_left = xyz[name_to_idx["plane_rear_left_wheel_link"]]
    rear_right = xyz[name_to_idx["plane_rear_right_wheel_link"]]

    y_axis = _normalize(left_tip - right_tip)
    front_mid = 0.5 * (front_left + front_right)
    rear_mid = 0.5 * (rear_left + rear_right)
    x_guess = front_mid - rear_mid
    x_axis = _normalize(x_guess - np.dot(x_guess, y_axis) * y_axis)
    z_axis = _normalize(np.cross(x_axis, y_axis))
    if z_axis[2] < 0.0:
        z_axis = -z_axis
    return np.column_stack([x_axis, y_axis, z_axis])


def _euler_xyz_to_rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    rx_mat = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cx, -sx],
            [0.0, sx, cx],
        ],
        dtype=np.float64,
    )
    ry_mat = np.array(
        [
            [cy, 0.0, sy],
            [0.0, 1.0, 0.0],
            [-sy, 0.0, cy],
        ],
        dtype=np.float64,
    )
    rz_mat = np.array(
        [
            [cz, -sz, 0.0],
            [sz, cz, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return rz_mat @ ry_mat @ rx_mat


def _get_warning_local_z_adjustment(box_name: str) -> float:
    n = str(box_name).lower()
    if "engine" in n:
        return float(WARNING_ENGINE_Z_OFFSET)
    if "landing_gear" in n or "front_landing_gear" in n or "gear" in n:
        return float(WARNING_LANDING_GEAR_Z_OFFSET)
    if "rear_wing" in n:
        return float(WARNING_REAR_WING_Z_OFFSET)
    if "wing" in n:
        return float(WARNING_WING_Z_OFFSET)
    return 0.0


def _pick_box_center_world(
    boxes: list[dict[str, Any]],
    filters: list[str],
    center_world: np.ndarray,
    R_target_from_center: np.ndarray,
) -> np.ndarray | None:
    fl = [str(x).lower() for x in filters]
    for b in boxes:
        n = str(b.get("name", "")).lower()
        if not any(f in n for f in fl):
            continue
        local = _box_center_local(b)
        return center_world + (R_target_from_center @ local)
    return None


def _pick_engine_lr_boxes(
    boxes: list[dict[str, Any]],
    center_world: np.ndarray | None = None,
    R_target_from_center: np.ndarray | None = None,
    ref_left_world: np.ndarray | None = None,
    ref_right_world: np.ndarray | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """
    Pick distinct engine-left / engine-right centers from warning boxes.

    Priority:
    1) Split all engine-like boxes by local-y sign (left: y>=0, right: y<0).
       Choose strongest side candidate by |y|.
    2) Fallback to name filters.
    3) Ensure left/right are not the same local point.
    """
    eng_like: list[dict[str, Any]] = []

    for b in boxes:
        n = str(b.get("name", "")).lower()
        if "engine" not in n:
            continue
        eng_like.append(b)

    left_box = None
    right_box = None

    if eng_like:
        left_candidates = [b for b in eng_like if float(b.get("y", 0.0)) >= 0.0]
        right_candidates = [b for b in eng_like if float(b.get("y", 0.0)) < 0.0]

        if left_candidates:
            left_box = max(left_candidates, key=lambda bb: abs(float(bb.get("y", 0.0))))
        if right_candidates:
            right_box = max(right_candidates, key=lambda bb: abs(float(bb.get("y", 0.0))))

    if left_box is None:
        # name-based fallback
        fl = [str(x).lower() for x in WARNING_ENGINE_LEFT_NAME_FILTERS]
        for b in boxes:
            n = str(b.get("name", "")).lower()
            if any(f in n for f in fl):
                left_box = b
                break

    if right_box is None:
        fl = [str(x).lower() for x in WARNING_ENGINE_RIGHT_NAME_FILTERS]
        for b in boxes:
            n = str(b.get("name", "")).lower()
            if any(f in n for f in fl):
                right_box = b
                break

    # If scene-level engine refs exist, choose side candidates nearest to those refs.
    if center_world is not None and R_target_from_center is not None:
        def _to_world(bb: dict[str, Any]) -> np.ndarray:
            p_local = np.array([float(bb["x"]), float(bb["y"]), float(bb["z"])], dtype=np.float64)
            return center_world + (R_target_from_center @ p_local)

        left_pool = [b for b in eng_like if float(b.get("y", 0.0)) >= 0.0]
        right_pool = [b for b in eng_like if float(b.get("y", 0.0)) < 0.0]
        if not left_pool:
            left_pool = eng_like[:]
        if not right_pool:
            right_pool = eng_like[:]

        if ref_left_world is not None and left_pool:
            left_box = min(left_pool, key=lambda bb: float(np.linalg.norm(_to_world(bb) - ref_left_world)))
        if ref_right_world is not None and right_pool:
            # avoid selecting the same box as left
            right_candidates = [b for b in right_pool if left_box is None or str(b.get("name", "")) != str(left_box.get("name", ""))]
            if not right_candidates:
                right_candidates = right_pool
            right_box = min(right_candidates, key=lambda bb: float(np.linalg.norm(_to_world(bb) - ref_right_world)))

    # never return duplicated side boxes
    if left_box is not None and right_box is not None:
        same_name = str(left_box.get("name", "")) == str(right_box.get("name", ""))
        same_xyz = (
            abs(float(left_box.get("x", 0.0)) - float(right_box.get("x", 0.0))) < 1e-6
            and abs(float(left_box.get("y", 0.0)) - float(right_box.get("y", 0.0))) < 1e-6
            and abs(float(left_box.get("z", 0.0)) - float(right_box.get("z", 0.0))) < 1e-6
        )
        if same_name or same_xyz:
            right_box = None
    return left_box, right_box


def _box_center_local(b: dict[str, Any]) -> np.ndarray:
    p = np.array([float(b["x"]), float(b["y"]), float(b["z"])], dtype=np.float64)
    p[2] += _get_warning_local_z_adjustment(str(b.get("name", "")))
    return p


def _local_to_world(center_world: np.ndarray, R_target_from_center: np.ndarray, p_local: np.ndarray) -> np.ndarray:
    return center_world + (R_target_from_center @ p_local)


def _refine_point_in_warning_box(
    aircraft_pts_world: np.ndarray,
    center_world: np.ndarray,
    R_target_from_center: np.ndarray,
    box: dict[str, Any] | None,
) -> np.ndarray | None:
    if box is None or aircraft_pts_world.size == 0:
        return None

    c = _box_center_local(box)
    R_box_local = _euler_xyz_to_rotation_matrix(
        float(box.get("rx", 0.0)),
        float(box.get("ry", 0.0)),
        float(box.get("rz", 0.0)),
    )
    hx = max(0.25, 0.5 * abs(float(box.get("sx", 0.0))))
    hy = max(0.25, 0.5 * abs(float(box.get("sy", 0.0))))
    hz = max(0.25, 0.5 * abs(float(box.get("sz", 0.0))))

    # world -> target-local
    rel = aircraft_pts_world - center_world.reshape(1, 3)
    local_pts = rel @ R_target_from_center
    local_box_pts = (local_pts - c.reshape(1, 3)) @ R_box_local

    def _inside(dx: float, dy: float, dz: float):
        return (
            (np.abs(local_box_pts[:, 0]) <= dx)
            & (np.abs(local_box_pts[:, 1]) <= dy)
            & (np.abs(local_box_pts[:, 2]) <= dz)
        )

    m = _inside(hx, hy, hz)
    if int(np.count_nonzero(m)) < ENGINE_BOX_SNAP_MIN_POINTS:
        s = max(1.0, float(ENGINE_BOX_SNAP_EXPAND_FACTOR))
        m = _inside(hx * s, hy * s, hz * s)

    pts = aircraft_pts_world[m]
    if pts.shape[0] == 0:
        return None

    # robust representative point
    return np.median(pts, axis=0)


def _pick_engine_lr_world(
    boxes: list[dict[str, Any]],
    center_world: np.ndarray,
    R_target_from_center: np.ndarray,
    aircraft_pts_world: np.ndarray | None = None,
    ref_left_world: np.ndarray | None = None,
    ref_right_world: np.ndarray | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    left_box, right_box = _pick_engine_lr_boxes(
        boxes,
        center_world=center_world,
        R_target_from_center=R_target_from_center,
        ref_left_world=ref_left_world,
        ref_right_world=ref_right_world,
    )

    p_left = None
    p_right = None
    if aircraft_pts_world is not None and ENGINE_BOX_SNAP_ENABLED:
        p_left = _refine_point_in_warning_box(aircraft_pts_world, center_world, R_target_from_center, left_box)
        p_right = _refine_point_in_warning_box(aircraft_pts_world, center_world, R_target_from_center, right_box)

    # Guard against over-drift: keep snap only if it remains near the warning-box center.
    if ENGINE_BOX_SNAP_MAX_DRIFT_M > 0:
        if p_left is not None and left_box is not None:
            c_left = _local_to_world(center_world, R_target_from_center, _box_center_local(left_box))
            if float(np.linalg.norm(p_left - c_left)) > float(ENGINE_BOX_SNAP_MAX_DRIFT_M):
                p_left = None
        if p_right is not None and right_box is not None:
            c_right = _local_to_world(center_world, R_target_from_center, _box_center_local(right_box))
            if float(np.linalg.norm(p_right - c_right)) > float(ENGINE_BOX_SNAP_MAX_DRIFT_M):
                p_right = None

    if p_left is None and left_box is not None:
        p_left = _local_to_world(center_world, R_target_from_center, _box_center_local(left_box))
    if p_right is None and right_box is not None:
        p_right = _local_to_world(center_world, R_target_from_center, _box_center_local(right_box))

    return p_left, p_right


def _warning_box_world_points_for_scene(
    h5_path: str,
    bag_name: str,
    scene_names_list: list[str],
    kps_model: np.ndarray,
    aircraft_pts_world: np.ndarray | None,
    profile_map: dict[str, dict[str, str]],
    yaml_cache: dict[Path, list[dict[str, Any]]],
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}

    profile = _infer_profile_from_h5_path(h5_path)
    row = _lookup_warning_profile_entry(profile_map, profile=profile, bag_name=bag_name) if profile_map else None

    yaml_path: Path | None = None
    if row is not None:
        ypath = str(row.get(WARNING_YAML_COLUMN, "") or "").strip()
        if ypath:
            p = Path(ypath).expanduser().resolve()
            if p.exists():
                yaml_path = p

    if yaml_path is None:
        yaml_path = _resolve_warning_yaml_for_bag(
            bag_name=bag_name,
            profile=profile,
        )
    if yaml_path is None:
        return out

    boxes_all = _load_yaml_crop_boxes(yaml_path, yaml_cache)
    if not boxes_all:
        return out

    boxes = [
        b for b in boxes_all
        if int(b.get("warning_level", -1)) == int(WARNING_TARGET_LEVEL)
    ]
    if not boxes:
        boxes = list(boxes_all)

    csv_box_names = _parse_csv_box_name_filters(row) if row is not None else []
    if csv_box_names:
        csv_filters = [str(x).strip().lower() for x in csv_box_names if str(x).strip()]
        filtered = [
            b for b in boxes
            if any(f in str(b.get("name", "")).lower() for f in csv_filters)
        ]
        if filtered:
            boxes = filtered

    name_to_idx = {str(n): i for i, n in enumerate(scene_names_list)}
    if WARNING_CENTER_KEYPOINT_NAME not in name_to_idx:
        return out

    center_world = np.asarray(kps_model[name_to_idx[WARNING_CENTER_KEYPOINT_NAME]], dtype=np.float64)
    R_target = _estimate_target_from_center_rotation_from_arrays(scene_names_list, kps_model)
    if R_target is None:
        return out
    origin_world = center_world + (R_target @ np.asarray(WARNING_CENTER_FRAME_OFFSET, dtype=np.float64))

    # optional scene references (if present) to disambiguate best engine boxes
    ref_left_world = None
    ref_right_world = None
    nm_el = find_alias(scene_names_list, ENGINE_LEFT_BOX_ALIASES)
    nm_er = find_alias(scene_names_list, ENGINE_RIGHT_BOX_ALIASES)
    if nm_el in name_to_idx:
        ref_left_world = np.asarray(kps_model[name_to_idx[nm_el]], dtype=np.float64)
    if nm_er in name_to_idx:
        ref_right_world = np.asarray(kps_model[name_to_idx[nm_er]], dtype=np.float64)

    p_front = _pick_box_center_world(boxes, WARNING_FRONT_GEAR_NAME_FILTERS, origin_world, R_target)
    if p_front is None and WARNING_DERIVE_FRONT_GEAR_FROM_WHEELS:
        nm_fr = find_alias(scene_names_list, FRONT_RIGHT_ALIASES)
        nm_fl = find_alias(scene_names_list, FRONT_LEFT_ALIASES)
        if nm_fr in name_to_idx and nm_fl in name_to_idx:
            p_front = 0.5 * (
                np.asarray(kps_model[name_to_idx[nm_fr]], dtype=np.float64)
                + np.asarray(kps_model[name_to_idx[nm_fl]], dtype=np.float64)
            )

    p_el, p_er = _pick_engine_lr_world(
        boxes,
        origin_world,
        R_target,
        aircraft_pts_world=aircraft_pts_world,
        ref_left_world=ref_left_world,
        ref_right_world=ref_right_world,
    )

    if p_front is not None:
        out["front_landing_gear"] = p_front
    if p_el is not None:
        out["engine_left"] = p_el
    if p_er is not None:
        out["engine_right"] = p_er
    return out


def _short_kp_label(name: str) -> str:
    m = {
        "plane_rear_left_wheel_link": "RLW",
        "plane_rear_right_wheel_link": "RRW",
        "left_wing_tip": "LWT",
        "right_wing_tip": "RWT",
        "front_wheels_mid": "FG",
        "engine_left_box_center": "EL",
        "engine_right_box_center": "ER",
    }
    if name in m:
        return m[name]
    parts = [p for p in str(name).replace('-', '_').split('_') if p]
    if not parts:
        return str(name)
    short = ''.join(s[0].upper() for s in parts[:3])
    return short if short else str(name)


def _xywhn_to_xyxy_clamped(
    cx: float,
    cy: float,
    bw: float,
    bh: float,
    W: int,
    H: int,
) -> tuple[int, int, int, int]:
    x1 = int(round((float(cx) - 0.5 * float(bw)) * float(W)))
    y1 = int(round((float(cy) - 0.5 * float(bh)) * float(H)))
    x2 = int(round((float(cx) + 0.5 * float(bw)) * float(W)))
    y2 = int(round((float(cy) + 0.5 * float(bh)) * float(H)))
    x1 = int(np.clip(x1, 0, max(0, W - 1)))
    y1 = int(np.clip(y1, 0, max(0, H - 1)))
    x2 = int(np.clip(x2, 0, max(0, W - 1)))
    y2 = int(np.clip(y2, 0, max(0, H - 1)))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def _parse_pose_label_text_for_viz(
    label_text: str,
    kp_order: list[str],
    H: int,
    W: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ln in str(label_text).splitlines():
        s = ln.strip()
        if not s:
            continue
        toks = s.split()
        if len(toks) < 5:
            continue
        try:
            cls_id = int(float(toks[0]))
            cx, cy, bw, bh = [float(x) for x in toks[1:5]]
        except Exception:
            continue

        x1, y1, x2, y2 = _xywhn_to_xyxy_clamped(cx, cy, bw, bh, W=W, H=H)
        kps: dict[str, tuple[int, int]] = {}
        rem = toks[5:]
        if len(rem) % 3 == 0 and kp_order:
            nk = min(len(kp_order), len(rem) // 3)
            for i in range(nk):
                try:
                    xn = float(rem[3 * i + 0])
                    yn = float(rem[3 * i + 1])
                    vv = float(rem[3 * i + 2])
                except Exception:
                    continue
                if vv <= 0:
                    continue
                c = int(np.clip(round(xn * float(W)), 0, max(0, W - 1)))
                r = int(np.clip(round(yn * float(H)), 0, max(0, H - 1)))
                kps[kp_order[i]] = (r, c)

        rows.append(
            {
                "cls_id": cls_id,
                "bbox": (x1, y1, x2, y2),
                "kps": kps,
            }
        )
    return rows


def _render_vis_from_label_text(
    img: np.ndarray,
    label_text: str,
    kp_order: list[str],
    split_name: str,
    unique_scene: str,
) -> np.ndarray:
    vis_img = img.copy()
    if vis_img.ndim == 2:
        vis_img = np.dstack([vis_img, vis_img, vis_img])
    H, W = vis_img.shape[:2]

    rows = _parse_pose_label_text_for_viz(label_text=label_text, kp_order=kp_order, H=H, W=W)
    for row in rows:
        x1, y1, x2, y2 = row["bbox"]
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for kp_name, (r0, c0) in row["kps"].items():
            cv2.circle(vis_img, (int(c0), int(r0)), 4, (255, 255, 255), -1, lineType=cv2.LINE_AA)
            cv2.circle(vis_img, (int(c0), int(r0)), 4, (0, 0, 0), 1, lineType=cv2.LINE_AA)
            cv2.putText(
                vis_img,
                _short_kp_label(kp_name),
                (int(c0) + 3, int(r0) - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    primary_kps = rows[0]["kps"] if rows else {}
    if DRAW_ENGINE_VIS_BBOX:
        def _draw_engine_vis_box(kp_name: str, color: tuple[int, int, int]):
            if kp_name not in primary_kps:
                return
            r_e, c_e = primary_kps[kp_name]
            x1e = max(0, int(c_e) - int(ENGINE_VIS_BBOX_HALF_W))
            y1e = max(0, int(r_e) - int(ENGINE_VIS_BBOX_HALF_H))
            x2e = min(W - 1, int(c_e) + int(ENGINE_VIS_BBOX_HALF_W))
            y2e = min(H - 1, int(r_e) + int(ENGINE_VIS_BBOX_HALF_H))
            cv2.rectangle(vis_img, (x1e, y1e), (x2e, y2e), color, 2)

        _draw_engine_vis_box(ENGINE_LEFT_KP_NAME, (255, 200, 0))
        _draw_engine_vis_box(ENGINE_RIGHT_KP_NAME, (0, 200, 255))

    if DRAW_NOSE_GEAR_VIS_BBOX:
        nose_candidates = []
        if USE_WARNING_BOX_KEYPOINTS:
            nose_candidates.append("front_landing_gear")
        nose_candidates.extend(NOSE_GEAR_ALIASES)
        nose_candidates.append(SYN_KP_NAME)
        for nose_name in nose_candidates:
            if nose_name not in primary_kps:
                continue
            r_n, c_n = primary_kps[nose_name]
            x1n = max(0, int(c_n) - int(NOSE_VIS_BBOX_HALF_W))
            y1n = max(0, int(r_n) - int(NOSE_VIS_BBOX_HALF_H))
            x2n = min(W - 1, int(c_n) + int(NOSE_VIS_BBOX_HALF_W))
            y2n = min(H - 1, int(r_n) + int(NOSE_VIS_BBOX_HALF_H))
            cv2.rectangle(vis_img, (x1n, y1n), (x2n, y2n), (255, 0, 255), 2)
            cv2.putText(
                vis_img,
                "NG",
                (x1n + 2, max(10, y1n - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 0, 255),
                1,
                cv2.LINE_AA,
            )
            break

    cv2.putText(
        vis_img,
        f"{split_name}/{unique_scene}",
        (5, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return vis_img


def _write_ascii_ply_xyz_rgb(xyz: np.ndarray, rgb_u8: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = int(xyz.shape[0])
    with out_path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            x, y, z = xyz[i]
            r, g, b = rgb_u8[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")


def _kp_marker_points(center_xyz: np.ndarray, scale: float = 0.22) -> np.ndarray:
    # Cross + diagonals so markers remain visible in dense clouds.
    c = np.asarray(center_xyz, dtype=np.float64).reshape(1, 3)
    offs = np.array(
        [
            [0, 0, 0],
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1],
            [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
            [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
            [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1],
        ],
        dtype=np.float64,
    )
    return c + float(scale) * offs


def _save_debug_pointcloud_with_keypoints(
    *,
    out_dir: Path,
    split_name: str,
    unique_scene: str,
    aircraft_pts_world: np.ndarray,
    kp3d_by_name: dict[str, np.ndarray],
    max_points: int,
) -> None:
    if aircraft_pts_world.size == 0:
        return

    pts = np.asarray(aircraft_pts_world, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] > int(max_points):
        idx = np.random.choice(pts.shape[0], size=int(max_points), replace=False)
        pts = pts[idx]

    base_rgb = np.full((pts.shape[0], 3), 145, dtype=np.uint8)
    xyz_parts = [pts]
    rgb_parts = [base_rgb]

    palette = np.array(
        [
            [255, 50, 50],
            [50, 220, 50],
            [50, 120, 255],
            [255, 180, 40],
            [255, 70, 220],
            [40, 230, 230],
            [255, 255, 70],
            [180, 120, 255],
            [255, 140, 110],
            [120, 255, 170],
        ],
        dtype=np.uint8,
    )

    legend_lines = []
    for i, nm in enumerate(sorted(kp3d_by_name.keys())):
        p = np.asarray(kp3d_by_name[nm], dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(p)):
            continue
        color = palette[i % len(palette)]
        marks = _kp_marker_points(p, scale=0.22)
        xyz_parts.append(marks)
        rgb_parts.append(np.tile(color.reshape(1, 3), (marks.shape[0], 1)))
        legend_lines.append(
            f"{nm}: xyz=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f}) rgb=({int(color[0])},{int(color[1])},{int(color[2])})"
        )

    xyz_all = np.vstack(xyz_parts)
    rgb_all = np.vstack(rgb_parts)

    ply_dir = out_dir / "debug_pointcloud_kps" / split_name
    ply_path = ply_dir / f"{unique_scene}.ply"
    _write_ascii_ply_xyz_rgb(xyz_all, rgb_all, ply_path)

    if legend_lines:
        (ply_dir / f"{unique_scene}.txt").write_text("\n".join(legend_lines) + "\n", encoding="utf-8")


def _show_debug_pointcloud_with_keypoints(
    *,
    unique_scene: str,
    aircraft_pts_world: np.ndarray,
    kp3d_by_name: dict[str, np.ndarray],
    max_points: int,
) -> bool:
    try:
        import open3d as o3d
    except Exception as e:
        print(f"[debug3d] Open3D unavailable: {e}")
        return False

    if aircraft_pts_world.size == 0:
        return False

    pts = np.asarray(aircraft_pts_world, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] > int(max_points):
        idx = np.random.choice(pts.shape[0], size=int(max_points), replace=False)
        pts = pts[idx]

    base_rgb = np.full((pts.shape[0], 3), 145, dtype=np.float64) / 255.0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(base_rgb)

    palette = np.array(
        [
            [255, 50, 50],
            [50, 220, 50],
            [50, 120, 255],
            [255, 180, 40],
            [255, 70, 220],
            [40, 230, 230],
            [255, 255, 70],
            [180, 120, 255],
            [255, 140, 110],
            [120, 255, 170],
        ],
        dtype=np.float64,
    ) / 255.0

    geoms: list[object] = [pcd]
    for i, nm in enumerate(sorted(kp3d_by_name.keys())):
        p = np.asarray(kp3d_by_name[nm], dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(p)):
            continue
        color = palette[i % len(palette)]
        marks = _kp_marker_points(p, scale=0.22)
        m = o3d.geometry.PointCloud()
        m.points = o3d.utility.Vector3dVector(marks)
        m.colors = o3d.utility.Vector3dVector(np.tile(color.reshape(1, 3), (marks.shape[0], 1)))
        geoms.append(m)

    o3d.visualization.draw_geometries(
        geoms,
        window_name=f"KP Debug: {unique_scene}",
        width=1400,
        height=900,
    )
    return True

def _balanced_bag_split_by_scene_count(
    scenes_by_file: Dict[str, List[str]],
    split: Tuple[float, float, float],
    seed: int,
):
    """
    Assign whole H5 files to train/val/test to approximately match scene-count ratios.
    Returns:
      split_for_file(h5p)->"train"/"val"/"test",
      assigned_files dict,
      scene_counts dict,
      targets dict
    """
    rng = random.Random(seed)

    items = [(h5p, len(scene_list)) for h5p, scene_list in scenes_by_file.items()]
    rng.shuffle(items)

    total_scenes_all = sum(n for _, n in items)
    if total_scenes_all == 0:
        raise RuntimeError("No scenes found for splitting.")

    # targets in scenes
    target_train = int(total_scenes_all * split[0])
    target_val = int(total_scenes_all * split[1])
    target_test = total_scenes_all - target_train - target_val

    targets = {"train": target_train, "val": target_val, "test": target_test}

    assigned = {"train": set(), "val": set(), "test": set()}
    count = {"train": 0, "val": 0, "test": 0}

    # Greedy: place largest files first where deficit is largest
    for h5p, nsc in sorted(items, key=lambda x: x[1], reverse=True):
        deficits = {k: targets[k] - count[k] for k in count}
        best = max(deficits, key=lambda k: deficits[k])
        assigned[best].add(h5p)
        count[best] += nsc

    def split_for_file(h5p: str) -> str:
        if h5p in assigned["train"]:
            return "train"
        if h5p in assigned["val"]:
            return "val"
        return "test"

    return split_for_file, assigned, count, targets


def create_dataset(
    source: str,
    out_dir: str,
    split: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    max_h5_files: Optional[int] = None,
) -> None:
    """
    Create YOLO-pose dataset from aircraft HDF5 LiDAR range-image data.

    Args:
        source (str): root directory or gs:// bucket
        out_dir (str): output dataset folder
        split (tuple): (train, val, test) ratios
        max_h5_files (int or None): limit number of HDF5 files
    """

    if abs(sum(split) - 1.0) >= 1e-6:
        raise ValueError(f"split must sum to 1.0 (got {sum(split):.4f})")

    random.seed(RANDOM_SEED)
    out_root = Path(out_dir)
    for p in [
        "images/train",
        "images/val",
        "images/test",
        "labels/train",
        "labels/val",
        "labels/test",
    ]:
        (out_root / p).mkdir(parents=True, exist_ok=True)

    # vis root
    vis_root = out_root / "vis"
    for p in ["train", "val", "test"]:
        (vis_root / p).mkdir(parents=True, exist_ok=True)

    print("[list] Searching for .h5 files…")
    h5_paths = list_h5_paths(source)
    if not h5_paths:
        raise RuntimeError(f"No .h5 files found under: {source}")

    print(f"[list] Found {len(h5_paths)} HDF5 files total.")

    # Limit number for testing
    if max_h5_files is not None and max_h5_files > 0 and len(h5_paths) > max_h5_files:
        h5_paths = h5_paths[:max_h5_files]
        print(f"[list] Using only first {len(h5_paths)} HDF5 files (max_h5_files={max_h5_files})")

    # ==================================================
    # Phase 1 — scan and build unified KEYPOINT ORDER
    # ==================================================

    all_scenes: List[Tuple[str, str]] = []
    KP_ORDER: List[str] = []
    seen_kps = set()
    per_scene_names: Dict[Tuple[str, str], List[str]] = {}

    print("\n--- Phase 1: Scanning files for scenes/keypoints ---")
    for i, h5p in enumerate(h5_paths, 1):
        print(f"[{i}/{len(h5_paths)}] {Path(h5p).name}")
        try:
            with open_h5_any(h5p) as f:
                for s, g in f.items():
                    if isinstance(g, h5py.Group) and "points" in g and "keypoints" in g:
                        all_scenes.append((h5p, s))
                        raw = g["keypoints"]["names"][()]
                        names = [
                            n.decode("utf-8") if isinstance(n, (bytes, bytearray)) else str(n)
                            for n in raw
                        ]
                        per_scene_names[(h5p, s)] = names
                        for n in names:
                            if n in REMOVE_KP_SET:
                                continue
                            if n not in seen_kps:
                                KP_ORDER.append(n)
                                seen_kps.add(n)
        except Exception as e:
            print(f"  [WARN] Skipping file due to error: {e}")

    if SYN_KP_NAME not in KP_ORDER:
        KP_ORDER.append(SYN_KP_NAME)

    if ADD_ENGINE_WARNING_BOX_KEYPOINTS:
        if ENGINE_LEFT_KP_NAME not in KP_ORDER:
            KP_ORDER.append(ENGINE_LEFT_KP_NAME)
        if ENGINE_RIGHT_KP_NAME not in KP_ORDER:
            KP_ORDER.append(ENGINE_RIGHT_KP_NAME)

    if not all_scenes:
        raise RuntimeError("No valid scenes found.")

    print(f"\n[index] Scenes: {len(all_scenes)}")
    print(f"[kps] Unified KP_ORDER ({len(KP_ORDER)}): {', '.join(KP_ORDER)}")

    # Group scenes by file
    scenes_by_file: Dict[str, List[str]] = defaultdict(list)
    for h5p, s in all_scenes:
        scenes_by_file[h5p].append(s)

    # ==================================================
    # Bag-level split (balanced by scene counts)
    # ==================================================
    split_for_file, assigned_files, scene_counts, targets = _balanced_bag_split_by_scene_count(
        scenes_by_file=scenes_by_file,
        split=split,
        seed=RANDOM_SEED,
    )
    write_bag_split_columns_csv(
        out_dir=out_dir,
        assigned_files=assigned_files,
        filename="bag_split_columns.csv",
    )

    print("\n--- Split (BAG-LEVEL, balanced by SCENE COUNT) ---")
    print("[split] target scenes:", targets)
    print("[split] actual scenes:", scene_counts)
    print("[split] files per split:", {k: len(v) for k, v in assigned_files.items()})

    # ==================================================
    # Phase 2 — export dataset
    # ==================================================

    print("\n--- Phase 2: Exporting images/labels (grouped by file) ---")
    total_files = len(scenes_by_file)

    warning_profile_map = _load_warning_profile_map(WARNING_PROFILE_CSV) if USE_WARNING_BOX_KEYPOINTS else {}
    yaml_boxes_cache: dict[Path, list[dict[str, Any]]] = {}

    total_valid_scenes = 0
    files_with_valid_scenes = 0
    debug_pc_saved = 0
    debug_pc_seen = 0
    debug_view_shown = 0
    debug_view_seen = 0
    debug_live_available = True
    reuse_root = _resolve_reuse_root(REUSE_LABELS_FROM_DIR)
    reuse_index: dict[str, list[Path]] = {}
    reuse_written = 0
    reuse_remapped = 0
    reuse_missing = 0
    reuse_invalid = 0
    reuse_strict_skipped = 0
    reuse_source_kp_names: list[str] = []
    if reuse_root is not None:
        if not reuse_root.exists():
            msg = f"[reuse] REUSE_LABELS_FROM_DIR does not exist: {reuse_root}"
            if REUSE_LABELS_STRICT:
                raise RuntimeError(msg)
            print(msg + " (fallback to generated labels)")
            reuse_root = None
        else:
            reuse_index = _build_reuse_label_index(reuse_root)
            yaml_candidates = [
                reuse_root / "aircraft_pose.yaml",
                reuse_root.parent / "aircraft_pose.yaml",
            ]
            for yp in yaml_candidates:
                reuse_source_kp_names = _load_yaml_keypoint_names(yp)
                if reuse_source_kp_names:
                    print(
                        f"[reuse] source keypoints from {yp} "
                        f"(count={len(reuse_source_kp_names)})"
                    )
                    break
            print(
                f"[reuse] enabled from {reuse_root} "
                f"(indexed stems: {len(reuse_index)}, strict={bool(REUSE_LABELS_STRICT)})"
            )
    else:
        print("[reuse] disabled (REUSE_LABELS_FROM_DIR empty)")

    for fi, (h5p, scene_list) in enumerate(scenes_by_file.items(), 1):
        split_name_for_file = split_for_file(h5p)
        print(f"[{fi}/{total_files}] {Path(h5p).name}  scenes={len(scene_list)}  -> {split_name_for_file}")

        file_valid_scenes = 0

        try:
            with open_h5_any(h5p) as f:
                H = int(f.attrs["height"])
                W = int(f.attrs["width"])

                for scene_name in scene_list:
                    file_stem = Path(h5p).stem
                    unique_scene = f"{file_stem}__{scene_name}"
                    split_name = split_name_for_file
                    print(f"  - {unique_scene} → {split_name}")
                    reused_label_text: str | None = None

                    grp = f[scene_name]
                    ds = grp["points"]
                    flat = ds[()]  # (H*W, C)

                    # --- columns ---
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
                    if not {"x", "y", "z"}.issubset(set(cols)):
                        print("    [SKIP] Missing x/y/z")
                        continue
                    ix, iy, iz = cols.index("x"), cols.index("y"), cols.index("z")

                    if "is_aircraft" not in cols:
                        print("    [SKIP] No is_aircraft")
                        continue
                    mask2d = (
                        flat[:, cols.index("is_aircraft")]
                        .astype(np.uint8)
                        .reshape(H, W)
                        .astype(bool)
                    )
                    ground2d = None
                    if "is_ground" in cols:
                        ground2d = (
                            flat[:, cols.index("is_ground")]
                            .astype(np.uint8)
                            .reshape(H, W)
                            .astype(bool)
                        )

                    # --- image ---
                    rgb = build_rgb_from_cols(flat, cols, H, W)
                    gray = build_gray_from_cols(flat, cols, H, W) if EXPORT_SINGLE_CHANNEL_IMAGE else None
                    if rgb is None:
                        gray_fallback = (mask2d.astype(np.uint8) * 255)
                        rgb = np.dstack([gray_fallback, gray_fallback, gray_fallback])
                    if EXPORT_SINGLE_CHANNEL_IMAGE:
                        if gray is None:
                            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                        img = gray
                    else:
                        img = rgb

                    if GROUND_SEPARATION_ENABLE and (ground2d is not None):
                        ground_only = ground2d & (~mask2d)
                        if np.any(ground_only):
                            att = float(np.clip(GROUND_ATTENUATION_FACTOR, 0.0, 1.0))
                            if img.ndim == 2:
                                img = img.copy()
                                img[ground_only] = np.round(
                                    img[ground_only].astype(np.float32) * att
                                ).astype(np.uint8)
                            else:
                                img = img.copy()
                                img[ground_only] = np.round(
                                    img[ground_only].astype(np.float32) * att
                                ).astype(np.uint8)

                    # optional overlay
                    if DRAW_ON_OVERLAY:
                        alpha = 0.5
                        if img.ndim == 2:
                            overlay = img.copy().astype(np.float32)
                            overlay[mask2d] = alpha * 255.0 + (1.0 - alpha) * overlay[mask2d]
                            img = np.clip(overlay, 0.0, 255.0).astype(np.uint8)
                        else:
                            overlay = img.copy()
                            red = np.zeros_like(overlay)
                            red[..., 0] = 255
                            overlay[mask2d] = (
                                alpha * red[mask2d] + (1 - alpha) * overlay[mask2d]
                            ).astype(np.uint8)
                            img = overlay

                    # optional median filter
                    if APPLY_MEDIAN_FILTER:
                        if img.dtype != np.uint8:
                            img = img.astype(np.uint8)
                        img = cv2.medianBlur(img, MEDIAN_KSIZE)

                    # xyz grid (LiDAR frame)
                    xyz = np.stack([flat[:, ix], flat[:, iy], flat[:, iz]], axis=1).astype(np.float64)
                    xyz_hw3 = xyz.reshape(H, W, 3)

                    # --- bbox from is_aircraft mask (BEFORE possible roll) ---
                    bb = bbox_from_mask(mask2d)
                    if bb is None:
                        print("    [SKIP] Empty aircraft mask")
                        continue
                    x1, y1, x2, y2 = bb
                    bbox_w = (x2 - x1 + 1)
                    bbox_frac = bbox_w / float(W)

                    # --- wrap-aware bbox (shortest circular arc) ---
                    if ROLL_WIDE_BBOX and W > 1:
                        bb_wrap, shift_wrap = _wrap_aware_bbox(mask2d)
                        if bb_wrap is None:
                            print("    [SKIP] Empty aircraft mask in wrap-aware bbox")
                            continue
                        if shift_wrap != 0:
                            x1w, y1w, x2w, y2w = bb_wrap
                            bw_wrap = (x2w - x1w + 1)
                            frac_wrap = bw_wrap / float(W)
                            print(f"    [ROLL-WRAP] bbox {bbox_frac:.3f}->{frac_wrap:.3f} by {shift_wrap} cols")
                            img = np.roll(img, shift=shift_wrap, axis=1)
                            mask2d = np.roll(mask2d, shift=shift_wrap, axis=1)
                            xyz_hw3 = np.roll(xyz_hw3, shift=shift_wrap, axis=1)
                            x1, y1, x2, y2 = x1w, y1w, x2w, y2w
                            bbox_w = bw_wrap
                            bbox_frac = frac_wrap

                    # --- ROLL LOGIC (smart seam placement) ---
                    if ROLL_WIDE_BBOX and bbox_frac > ROLL_WIDE_BBOX_FRAC and W > 1:
                        shift = find_best_azimuth_roll(mask2d)
                        if shift == 0:
                            shift = ROLL_WIDE_BBOX_COLS % W
                        if shift != 0:
                            print(f"    [ROLL] Wide bbox (frac={bbox_frac:.3f}) → rolling by {shift} cols")
                            img = np.roll(img, shift=shift, axis=1)
                            mask2d = np.roll(mask2d, shift=shift, axis=1)
                            xyz_hw3 = np.roll(xyz_hw3, shift=shift, axis=1)

                            bb2 = bbox_from_mask(mask2d)
                            if bb2 is not None:
                                x1, y1, x2, y2 = bb2
                                bbox_w = (x2 - x1 + 1)
                                bbox_frac = bbox_w / float(W)

                    if bbox_frac > 0.6:
                        print(f"    [SKIP] BBox too wide ({bbox_frac:.3f} > 0.6) in {unique_scene}")
                        continue

                    # normalized bbox
                    cx, cy, bw, bh = xyxy_to_xywhn(x1, y1, x2, y2, W, H)

                    # Optional label reuse from external dataset/folder.
                    if reuse_root is not None:
                        reuse_label_path = _pick_reuse_label_path(
                            reuse_root=reuse_root,
                            reuse_index=reuse_index,
                            split_name=split_name,
                            unique_scene=unique_scene,
                        )
                        if reuse_label_path is None:
                            reuse_missing += 1
                            if REUSE_LABELS_STRICT:
                                reuse_strict_skipped += 1
                                print("    [SKIP] Reuse strict: no external label found")
                                continue
                        else:
                            try:
                                raw_reuse = reuse_label_path.read_text(
                                    encoding="utf-8", errors="ignore"
                                )
                                reused_label_text, reuse_err, reuse_was_remapped = _prepare_reuse_pose_label_text(
                                    raw_reuse,
                                    target_kp_names=KP_ORDER,
                                    source_kp_names=reuse_source_kp_names,
                                )
                                if reused_label_text is None:
                                    reuse_invalid += 1
                                    print(
                                        f"    [REUSE-WARN] Invalid external label "
                                        f"({reuse_label_path.name}): {reuse_err}"
                                    )
                                    if REUSE_LABELS_STRICT:
                                        reuse_strict_skipped += 1
                                        print("    [SKIP] Reuse strict: external label invalid")
                                        continue
                                else:
                                    if reuse_was_remapped:
                                        reuse_remapped += 1
                                        print(f"    [REUSE] remapped label from: {reuse_label_path}")
                                    else:
                                        print(f"    [REUSE] label: {reuse_label_path}")
                            except Exception as e:
                                reuse_invalid += 1
                                print(f"    [REUSE-WARN] Failed reading external label: {e}")
                                if REUSE_LABELS_STRICT:
                                    reuse_strict_skipped += 1
                                    print("    [SKIP] Reuse strict: failed reading external label")
                                    continue

                    # aircraft points and ground z
                    aircraft_pts = xyz_hw3[mask2d]  # (Na, 3)
                    z_min_air = float(np.min(aircraft_pts[:, 2])) if aircraft_pts.size > 0 else 0.0

                    # range image for ray-tracing visibility
                    range_img = np.linalg.norm(xyz_hw3, axis=2)  # (H, W)
                    valid_range = np.isfinite(range_img) & mask2d

                    # Vectorized elevation/azimuth calibration
                    valid = np.all(np.isfinite(xyz_hw3), axis=2)
                    el_all = np.full((H, W), np.nan, dtype=np.float64)
                    az_all = np.full((H, W), np.nan, dtype=np.float64)
                    xv = xyz_hw3[..., 0][valid]
                    yv = xyz_hw3[..., 1][valid]
                    zv = xyz_hw3[..., 2][valid]
                    el_all[valid] = np.arctan2(zv, np.sqrt(xv * xv + yv * yv))
                    az_all[valid] = np.arctan2(yv, xv)
                    el_per_row_raw = np.nanmedian(el_all, axis=1)  # (H,)
                    sin_c = np.nanmean(np.sin(az_all), axis=0)
                    cos_c = np.nanmean(np.cos(az_all), axis=0)
                    az_per_col_raw = np.arctan2(sin_c, cos_c)  # (W,)
                    el_per_row_calib = _fill_nans(el_per_row_raw)
                    az_per_col_calib = _fill_nans(az_per_col_raw)

                    # keypoints from file
                    kp_grp = grp["keypoints"]
                    kps_model = np.asarray(kp_grp["xyz"][()], dtype=np.float64)
                    raw_names = kp_grp.get("names", None)
                    scene_names_list = [
                        n.decode("utf-8") if isinstance(n, (bytes, bytearray)) else str(n)
                        for n in (raw_names[()] if raw_names is not None else [])
                    ]
                    if kps_model.ndim != 2 or kps_model.shape[1] != 3 or kps_model.shape[0] == 0:
                        print("    [SKIP] Keypoints malformed")
                        continue

                    ok_rows = np.all(np.isfinite(kps_model), axis=1)
                    kps_model = kps_model[ok_rows]
                    if scene_names_list:
                        scene_names_list = [scene_names_list[i] for i, t in enumerate(ok_rows) if t]
                    else:
                        scene_names_list = [f"k{i}" for i in range(kps_model.shape[0])]

                    name_to_idx_full = {n: i for i, n in enumerate(scene_names_list)}

                    warning_world_pts = {}
                    if USE_WARNING_BOX_KEYPOINTS:
                        warning_world_pts = _warning_box_world_points_for_scene(
                            h5_path=h5p,
                            bag_name=file_stem,
                            scene_names_list=scene_names_list,
                            kps_model=kps_model,
                            aircraft_pts_world=aircraft_pts,
                            profile_map=warning_profile_map,
                            yaml_cache=yaml_boxes_cache,
                        )

                    # Synthetic front point source.
                    # When warning-box merger is enabled, use only merged front-gear.
                    # Otherwise, use legacy nose-gear/midpoint fallbacks.
                    mid_raw_3d = None
                    mid_from_nose_gear = False

                    if USE_WARNING_BOX_KEYPOINTS:
                        if "front_landing_gear" in warning_world_pts:
                            mid_raw_3d = warning_world_pts["front_landing_gear"]
                            mid_from_nose_gear = True
                    else:
                        if USE_NOSE_GEAR_CENTER_FOR_SYNTHETIC_FRONT_MID:
                            nm_nose = find_alias(scene_names_list, NOSE_GEAR_ALIASES)
                            if nm_nose in name_to_idx_full:
                                mid_raw_3d = kps_model[name_to_idx_full[nm_nose]]
                                mid_from_nose_gear = True

                        if mid_raw_3d is None:
                            nm_fr = find_alias(scene_names_list, FRONT_RIGHT_ALIASES)
                            nm_fl = find_alias(scene_names_list, FRONT_LEFT_ALIASES)
                            if (nm_fr in name_to_idx_full) and (nm_fl in name_to_idx_full):
                                p_fr = kps_model[name_to_idx_full[nm_fr]]
                                p_fl = kps_model[name_to_idx_full[nm_fl]]
                                mid_raw_3d = 0.5 * (p_fr + p_fl)

                    # remove unwanted KPs
                    keep_mask = [n not in REMOVE_KP_SET for n in scene_names_list]
                    names_kept = [n for n, keep in zip(scene_names_list, keep_mask) if keep]
                    kps_kept = kps_model[keep_mask]
                    name_to_idx_kept = {n: i for i, n in enumerate(names_kept)}

                    # apply transform only to base_link
                    T = None
                    if USE_TF_MATRIX and "metadata" in grp and "tf_matrix" in grp["metadata"]:
                        T_ = np.asarray(grp["metadata"]["tf_matrix"][()], dtype=np.float64)
                        if T_.shape == (4, 4) and np.all(np.isfinite(T_)):
                            T = T_ if not APPLY_Z_FLIP else (T_ @ np.diag([1.0, 1.0, -1.0, 1.0]))

                    kps_scene = kps_kept.copy()
                    if ("base_link" in name_to_idx_kept) and (T is not None):
                        i_base = name_to_idx_kept["base_link"]
                        kps_scene[i_base : i_base + 1] = apply_transform(kps_scene[i_base : i_base + 1], T)

                    # project regular keypoints
                    rc_by_name: Dict[str, Tuple[int, int]] = {}
                    vis_by_name: Dict[str, int] = {}

                    if kps_scene.size:
                        az_kp, el_kp = angles_from_xyz(kps_scene)
                        for jj, nm in enumerate(names_kept):
                            elv = float(el_kp[jj])
                            azv = float(az_kp[jj])
                            r = row_from_elevation(elv, el_per_row_calib, H)
                            c = col_from_azimuth_global(azv, az_per_col_calib, W)

                            if r < 0 or r >= H or c < 0 or c >= W:
                                vis_by_name[nm] = 0
                                continue

                            r_int = int(r)
                            c_int = int(c)

                            if nm == "base_link" and r_int == 0:
                                vis_by_name[nm] = 0
                                continue

                            skip_ray_check = nm in RAY_VISIBILITY_EXEMPT_SET
                            if RAY_VISIBILITY_CHECK and not skip_ray_check:
                                R_hit = get_min_depth(range_img, valid_range, r_int, c_int, RAY_PATCH_RADIUS)
                                if R_hit is None and nm in RAY_REQUIRE_LOCAL_HIT_SET:
                                    vis_by_name[nm] = 0
                                    continue
                                R_kp = float(np.linalg.norm(kps_scene[jj]))
                                if R_hit is not None and np.isfinite(R_kp):
                                    if R_kp > R_hit + RAY_TOL:
                                        vis_by_name[nm] = 0
                                        continue

                            rc_by_name[nm] = (r_int, c_int)
                            vis_by_name[nm] = 1

                    p_el = None
                    p_er = None
                    # Engine box-center keypoints (optional)
                    if ADD_ENGINE_WARNING_BOX_KEYPOINTS:
                        def _snap_pixel_local(r0: int, c0: int, radius: int) -> tuple[int, int]:
                            rmin = max(0, r0 - radius)
                            rmax = min(H, r0 + radius + 1)
                            cmin = max(0, c0 - radius)
                            cmax = min(W, c0 + radius + 1)
                            sub_valid = valid_range[rmin:rmax, cmin:cmax]
                            if not np.any(sub_valid):
                                if ENGINE_PIXEL_SNAP_FALLBACK_TO_NEAREST:
                                    rr_all, cc_all = np.where(valid_range)
                                    if rr_all.size > 0:
                                        dr_all = rr_all.astype(np.float64) - float(r0)
                                        dc_all = cc_all.astype(np.float64) - float(c0)
                                        dist2_all = dr_all * dr_all + dc_all * dc_all
                                        k_all = int(np.argmin(dist2_all))
                                        d_min = float(np.sqrt(dist2_all[k_all]))
                                        if d_min <= float(ENGINE_PIXEL_SNAP_FALLBACK_MAX_DIST):
                                            return int(rr_all[k_all]), int(cc_all[k_all])
                                return r0, c0

                            rr, cc = np.where(sub_valid)
                            rr = rr + rmin
                            cc = cc + cmin

                            # Prefer close pixels; lightly prefer nearer surface.
                            dr = rr.astype(np.float64) - float(r0)
                            dc = cc.astype(np.float64) - float(c0)
                            dist2 = dr * dr + dc * dc
                            rng = range_img[rr, cc]
                            score = dist2 + 0.02 * rng
                            k = int(np.argmin(score))
                            return int(rr[k]), int(cc[k])

                        def _project_optional_point(dst_name: str, p3: np.ndarray | None):
                            if p3 is None:
                                vis_by_name[dst_name] = 0
                                return

                            az_e, el_e = angles_from_xyz(np.asarray(p3, dtype=np.float64).reshape(1, 3))
                            r_e = row_from_elevation(float(el_e[0]), el_per_row_calib, H)
                            c_e = col_from_azimuth_global(float(az_e[0]), az_per_col_calib, W)
                            if r_e < 0 or r_e >= H or c_e < 0 or c_e >= W:
                                vis_by_name[dst_name] = 0
                                return

                            r_int = int(r_e)
                            c_int = int(c_e)
                            if ENGINE_PIXEL_SNAP_ENABLED:
                                r_int, c_int = _snap_pixel_local(r_int, c_int, ENGINE_PIXEL_SNAP_RADIUS)
                            if ENGINE_PIXEL_ROW_BIAS != 0:
                                r_int = int(np.clip(r_int + ENGINE_PIXEL_ROW_BIAS, 0, H - 1))
                            skip_ray_check = dst_name in RAY_VISIBILITY_EXEMPT_SET
                            if RAY_VISIBILITY_CHECK and not skip_ray_check:
                                R_hit_e = get_min_depth(range_img, valid_range, r_int, c_int, RAY_PATCH_RADIUS)
                                R_kp_e = float(np.linalg.norm(p3))
                                if R_hit_e is not None and np.isfinite(R_kp_e) and R_kp_e > R_hit_e + RAY_TOL:
                                    vis_by_name[dst_name] = 0
                                    return

                            rc_by_name[dst_name] = (r_int, c_int)
                            vis_by_name[dst_name] = 1

                        if USE_WARNING_BOX_KEYPOINTS:
                            # Strict merger source for engine keypoints.
                            p_el = warning_world_pts.get("engine_left")
                            p_er = warning_world_pts.get("engine_right")
                        else:
                            nm_src = find_alias(scene_names_list, ENGINE_LEFT_BOX_ALIASES)
                            p_el = kps_model[name_to_idx_full[nm_src]] if nm_src in name_to_idx_full else None
                            nm_src = find_alias(scene_names_list, ENGINE_RIGHT_BOX_ALIASES)
                            p_er = kps_model[name_to_idx_full[nm_src]] if nm_src in name_to_idx_full else None

                        _project_optional_point(ENGINE_LEFT_KP_NAME, p_el)
                        _project_optional_point(ENGINE_RIGHT_KP_NAME, p_er)

                    mid_adj_3d = None
                    # Synthetic front_wheels_mid
                    if (mid_raw_3d is not None) and (aircraft_pts.size > 0):
                        if mid_from_nose_gear:
                            mid_adj_3d = mid_raw_3d
                        else:
                            mid_adj_3d, _ = adjust_midpoint_to_cluster(
                                mid_raw_3d,
                                aircraft_pts,
                                z_min_air,
                                base_radius=MID_BASE_RADIUS,
                                expand_radius=MID_EXPAND_RADIUS,
                                z_band=MID_Z_BAND,
                                min_points=MID_MIN_POINTS,
                            )

                        az_syn, el_syn = angles_from_xyz(mid_adj_3d.reshape(1, 3))
                        r_syn = row_from_elevation(float(el_syn[0]), el_per_row_calib, H)
                        c_syn = col_from_azimuth_global(float(az_syn[0]), az_per_col_calib, W)

                        if 0 <= r_syn < H and 0 <= c_syn < W:
                            r_syn_int = int(r_syn)
                            c_syn_int = int(c_syn)

                            if RAY_VISIBILITY_CHECK:
                                R_hit_syn = get_min_depth(range_img, valid_range, r_syn_int, c_syn_int, RAY_PATCH_RADIUS)
                                R_kp_syn = float(np.linalg.norm(mid_adj_3d))
                                if R_hit_syn is not None and np.isfinite(R_kp_syn):
                                    if R_kp_syn > R_hit_syn + RAY_TOL:
                                        vis_by_name[SYN_KP_NAME] = 0
                                    else:
                                        rc_by_name[SYN_KP_NAME] = (r_syn_int, c_syn_int)
                                        vis_by_name[SYN_KP_NAME] = 1
                                else:
                                    rc_by_name[SYN_KP_NAME] = (r_syn_int, c_syn_int)
                                    vis_by_name[SYN_KP_NAME] = 1
                            else:
                                vis_by_name[SYN_KP_NAME] = 0
                    else:
                        vis_by_name[SYN_KP_NAME] = 0

                    # Optional 3D debug: build once, then export and/or show live.
                    kp3d_by_name: dict[str, np.ndarray] | None = None
                    if DEBUG_POINTCLOUD_KEYPOINTS or DEBUG_POINTCLOUD_LIVE_VIEWER:
                        kp3d_by_name = {}
                        for jj, nm in enumerate(names_kept):
                            kp3d_by_name[nm] = np.asarray(kps_scene[jj], dtype=np.float64)
                        if mid_adj_3d is not None:
                            kp3d_by_name[SYN_KP_NAME] = np.asarray(mid_adj_3d, dtype=np.float64)
                        if p_el is not None:
                            kp3d_by_name[ENGINE_LEFT_KP_NAME] = np.asarray(p_el, dtype=np.float64)
                        if p_er is not None:
                            kp3d_by_name[ENGINE_RIGHT_KP_NAME] = np.asarray(p_er, dtype=np.float64)
                        if "front_landing_gear" in warning_world_pts:
                            kp3d_by_name["front_landing_gear"] = np.asarray(
                                warning_world_pts["front_landing_gear"], dtype=np.float64
                            )

                    if DEBUG_POINTCLOUD_KEYPOINTS and kp3d_by_name is not None:
                        debug_pc_seen += 1
                        stride = max(1, int(DEBUG_POINTCLOUD_SAMPLE_EVERY_N))
                        if (debug_pc_seen - 1) % stride == 0 and debug_pc_saved < int(DEBUG_POINTCLOUD_MAX_SCENES):
                            _save_debug_pointcloud_with_keypoints(
                                out_dir=Path(out_dir),
                                split_name=split_name,
                                unique_scene=unique_scene,
                                aircraft_pts_world=aircraft_pts,
                                kp3d_by_name=kp3d_by_name,
                                max_points=int(DEBUG_POINTCLOUD_MAX_POINTS),
                            )
                            debug_pc_saved += 1

                    if DEBUG_POINTCLOUD_LIVE_VIEWER and kp3d_by_name is not None and debug_live_available:
                        debug_view_seen += 1
                        stride = max(1, int(DEBUG_POINTCLOUD_SAMPLE_EVERY_N))
                        if (debug_view_seen - 1) % stride == 0 and debug_view_shown < int(DEBUG_POINTCLOUD_MAX_SCENES):
                            ok_view = _show_debug_pointcloud_with_keypoints(
                                unique_scene=unique_scene,
                                aircraft_pts_world=aircraft_pts,
                                kp3d_by_name=kp3d_by_name,
                                max_points=int(DEBUG_POINTCLOUD_MAX_POINTS),
                            )
                            if ok_view:
                                debug_view_shown += 1
                            else:
                                debug_live_available = False

                    # save image
                    img_path = Path(out_dir) / "images" / split_name / f"{unique_scene}.png"
                    imageio.imwrite(str(img_path), img, compress_level=1)

                    # YOLO label
                    if reused_label_text is not None:
                        label_text = reused_label_text
                        reuse_written += 1
                    else:
                        parts = ["0", f"{cx:.6f}", f"{cy:.6f}", f"{bw:.6f}", f"{bh:.6f}"]
                        for kp in KP_ORDER:
                            if kp in rc_by_name and vis_by_name.get(kp, 0) > 0:
                                r0, c0 = rc_by_name[kp]
                                xn, yn = rc_to_xy_norm(r0, c0, H, W)
                                parts += [f"{xn:.6f}", f"{yn:.6f}", "1"]
                            else:
                                parts += ["0.000000", "0.000000", "0"]
                        label_text = " ".join(parts) + "\n"

                    (Path(out_dir) / "labels" / split_name / f"{unique_scene}.txt").write_text(label_text)

                    # visualizations (rendered from final written label text)
                    if MAKE_VIZ:
                        vis_img = _render_vis_from_label_text(
                            img=img,
                            label_text=label_text,
                            kp_order=KP_ORDER,
                            split_name=split_name,
                            unique_scene=unique_scene,
                        )
                        vis_path = vis_root / split_name / f"{unique_scene}.png"
                        imageio.imwrite(str(vis_path), vis_img, compress_level=1)

                    file_valid_scenes += 1
                    total_valid_scenes += 1

        except Exception as e:
            print(f"[ERROR] {Path(h5p).name}: {e}")

        if file_valid_scenes > 0:
            files_with_valid_scenes += 1
            print(f"  -> valid scenes in this file: {file_valid_scenes}")
        else:
            print("  -> no valid scenes exported from this file")

    print(f"\n[summary] Total valid scenes exported: {total_valid_scenes}")
    print(f"[summary] H5 files with >= 1 valid scene: {files_with_valid_scenes}/{total_files}")
    if DEBUG_POINTCLOUD_KEYPOINTS:
        print(
            f"[summary] 3D debug clouds saved: {debug_pc_saved} "
            f"(seen={debug_pc_seen}, stride={max(1, int(DEBUG_POINTCLOUD_SAMPLE_EVERY_N))}, "
            f"cap={int(DEBUG_POINTCLOUD_MAX_SCENES)})"
        )
    if DEBUG_POINTCLOUD_LIVE_VIEWER:
        print(
            f"[summary] 3D live views shown: {debug_view_shown} "
            f"(seen={debug_view_seen}, stride={max(1, int(DEBUG_POINTCLOUD_SAMPLE_EVERY_N))}, "
            f"cap={int(DEBUG_POINTCLOUD_MAX_SCENES)})"
        )
    if reuse_root is not None:
        print(
            "[summary] Reused labels: "
            f"written={reuse_written}, remapped={reuse_remapped}, missing={reuse_missing}, "
            f"invalid={reuse_invalid}, strict_skipped={reuse_strict_skipped}"
        )

    # YAML
    yaml_text = (
        f"""# YOLO pose dataset — unified from multiple H5 files
path: {Path(out_dir).resolve()}
train: images/train
val: images/val
test: images/test

names: ["aircraft"]

kpt_shape: [{len(KP_ORDER)}, 3]
keypoints:
"""
        + "".join([f"  - {n}\n" for n in KP_ORDER])
    )
    (Path(out_dir) / "aircraft_pose.yaml").write_text(yaml_text)

    # diagnostics
    (Path(out_dir) / "export_config.json").write_text(
        json.dumps(
            {
                "source": source,
                "out_dir": out_dir,
                "split": split,
                "split_mode": "bag_level_balanced_by_scene_count",
                "assigned_files": {k: sorted(list(v)) for k, v in assigned_files.items()},
                "target_scene_counts": targets,
                "actual_scene_counts": scene_counts,
                "kp_order": KP_ORDER,
                "image_render_mode": IMAGE_RENDER_MODE,
                "image_channel_fields": list(IMAGE_CHANNEL_FIELDS),
                "image_colormap": IMAGE_COLORMAP,
                "range_channel_mode": BLUE_CHANNEL_MODE,
                "range_channel_gamma": BLUE_CHANNEL_GAMMA,
                "far_bright_boost_enable": FAR_BRIGHT_BOOST_ENABLE,
                "far_bright_boost_fields": list(FAR_BRIGHT_BOOST_FIELDS),
                "far_bright_boost_strength": FAR_BRIGHT_BOOST_STRENGTH,
                "intensity_boost_enable": INTENSITY_BOOST_ENABLE,
                "intensity_boost_gain": INTENSITY_BOOST_GAIN,
                "intensity_row_correction_enable": INTENSITY_ROW_CORRECTION_ENABLE,
                "intensity_row_correction_strength": INTENSITY_ROW_CORRECTION_STRENGTH,
                "intensity_row_correction_sigma_rows": INTENSITY_ROW_CORRECTION_SIGMA_ROWS,
                "intensity_row_correction_max_shift": INTENSITY_ROW_CORRECTION_MAX_SHIFT,
                "export_single_channel_image": EXPORT_SINGLE_CHANNEL_IMAGE,
                "single_channel_field": SINGLE_CHANNEL_FIELD,
                "ground_separation_enable": GROUND_SEPARATION_ENABLE,
                "ground_attenuation_factor": GROUND_ATTENUATION_FACTOR,
                "kpt_bbox_margin_px": KPT_BBOX_MARGIN_PX,
                "roll_wide_bbox": ROLL_WIDE_BBOX,
                "roll_wide_bbox_frac": ROLL_WIDE_BBOX_FRAC,
                "roll_wide_bbox_cols": ROLL_WIDE_BBOX_COLS,
                "max_h5_files": max_h5_files,
                "total_valid_scenes": total_valid_scenes,
                "files_with_valid_scenes": files_with_valid_scenes,
                "total_h5_files_seen": total_files,
                "reuse_labels_from_dir": str(reuse_root) if reuse_root is not None else "",
                "reuse_labels_strict": bool(REUSE_LABELS_STRICT),
                "reuse_labels_written": reuse_written,
                "reuse_labels_remapped": reuse_remapped,
                "reuse_labels_missing": reuse_missing,
                "reuse_labels_invalid": reuse_invalid,
                "reuse_labels_strict_skipped": reuse_strict_skipped,
                "reuse_source_keypoints_count": len(reuse_source_kp_names),
                "visibility_definition": (
                    "1 = ray-visible (in image bounds, not base_link on row 0, and not clearly behind aircraft surface "
                    "in a local patch, except configured ray-exempt keypoints; and for configured keys, "
                    "a local aircraft hit is required); "
                    "0 = not visible or occluded. Scenes with bbox_width / image_width > 0.6 are skipped."
                ),
                "ray_visibility_check": RAY_VISIBILITY_CHECK,
                "ray_tolerance_m": RAY_TOL,
                "ray_patch_radius": RAY_PATCH_RADIUS,
                "ray_require_local_hit_keypoints": sorted(RAY_REQUIRE_LOCAL_HIT_SET),
                "ray_visibility_exempt_keypoints": sorted(RAY_VISIBILITY_EXEMPT_SET),
                "mid_base_radius": MID_BASE_RADIUS,
                "mid_expand_radius": MID_EXPAND_RADIUS,
                "mid_z_band": MID_Z_BAND,
                "mid_min_points": MID_MIN_POINTS,
                "use_warning_box_keypoints": USE_WARNING_BOX_KEYPOINTS,
                "warning_profile_csv": WARNING_PROFILE_CSV if USE_WARNING_BOX_KEYPOINTS else "",
                "engine_box_snap_enabled": ENGINE_BOX_SNAP_ENABLED,
                "engine_box_snap_min_points": ENGINE_BOX_SNAP_MIN_POINTS,
                "engine_box_snap_expand_factor": ENGINE_BOX_SNAP_EXPAND_FACTOR,
                "engine_pixel_snap_enabled": ENGINE_PIXEL_SNAP_ENABLED,
                "engine_pixel_snap_radius": ENGINE_PIXEL_SNAP_RADIUS,
                "debug_pointcloud_keypoints": DEBUG_POINTCLOUD_KEYPOINTS,
                "debug_pointcloud_max_scenes": DEBUG_POINTCLOUD_MAX_SCENES,
                "debug_pointcloud_sample_every_n": DEBUG_POINTCLOUD_SAMPLE_EVERY_N,
                "debug_pointcloud_max_points": DEBUG_POINTCLOUD_MAX_POINTS,
                "debug_pointcloud_saved": debug_pc_saved,
                "debug_pointcloud_live_viewer": DEBUG_POINTCLOUD_LIVE_VIEWER,
                "debug_pointcloud_live_shown": debug_view_shown,
            },
            indent=2,
        )
    )

    print("\n✓ Dataset exported to", Path(out_dir).resolve())
    print("  -> YAML:", (Path(out_dir) / "aircraft_pose.yaml").resolve())
    if MAKE_VIZ:
        print("  -> Visualizations under:", (Path(out_dir) / "vis").resolve())
    if DEBUG_POINTCLOUD_KEYPOINTS:
        print("  -> 3D debug PLY under:", (Path(out_dir) / "debug_pointcloud_kps").resolve())
    if DEBUG_POINTCLOUD_LIVE_VIEWER:
        print("  -> 3D live viewer was enabled (close each window to continue).")


def parse_arguments():
    """CLI overrides for SOURCE / OUT_DIR / SPLIT / MAX_H5_FILES."""
    parser = argparse.ArgumentParser(
        description="Create aircraft YOLO-pose dataset from HDF5 LiDAR files."
    )

    parser.add_argument(
        "--source",
        type=str,
        default=SOURCE,
        help=f"Input H5 directory or gs:// path (default: {SOURCE})",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=OUT_DIR,
        help=f"Output directory (default: {OUT_DIR})",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=SPLIT[0],
        help="Train split ratio (default from config_dataset.py)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=SPLIT[1],
        help="Validation split ratio",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=SPLIT[2],
        help="Test split ratio",
    )
    parser.add_argument(
        "--max-h5",
        type=int,
        default=MAX_H5_FILES,
        help="Limit number of H5 files (default from config_dataset.py)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    split_tuple = (args.train_ratio, args.val_ratio, args.test_ratio)

    create_dataset(
        source=args.source,
        out_dir=args.out,
        split=split_tuple,
        max_h5_files=args.max_h5,
    )
