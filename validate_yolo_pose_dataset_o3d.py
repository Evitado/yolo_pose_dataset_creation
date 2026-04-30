#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Validate a YOLO-pose dataset against real H5 point clouds with Open3D.

What this script does:
1) Reads image/label pairs from dataset root (images/{train,val,test}, labels/{train,val,test})
2) Maps each sample stem '<h5_stem>__<scene_name>' to the matching H5 + scene
3) Backprojects label bbox + keypoints from 2D image pixels to 3D xyz
4) Opens Open3D viewer with:
   - full real scene point cloud
   - bbox-highlighted cloud region
   - keypoint spheres at 3D positions from labels
5) Optionally runs warning-box inside/outside checks with per-sample PASS/FAIL

Example:
  python validate_yolo_pose_dataset_o3d.py

Or override from CLI:
  python validate_yolo_pose_dataset_o3d.py \
      --dataset-root /home/femi/yolo_pose_dataset_creation/aircraft_pose_with_normalising_applied_multifield_only_3_2 \
      --h5-source /home/femi/Benchmarking_framework/Data/warning_b_test_h5 \
      --split val --max-samples 20 --shuffle
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from config_dataset import ROLL_WIDE_BBOX, ROLL_WIDE_BBOX_COLS, ROLL_WIDE_BBOX_FRAC
try:
    from config_dataset import (
        WARNING_PROFILE_CSV,
        WARNING_YAML_COLUMN,
        WARNING_YAML_ROOT,
        WARNING_YAML_RELPATH,
        WARNING_TARGET_LEVEL,
        WARNING_CENTER_FRAME_OFFSET,
    )
except Exception:
    WARNING_PROFILE_CSV = ""
    WARNING_YAML_COLUMN = "recommended_yaml"
    WARNING_YAML_ROOT = ""
    WARNING_YAML_RELPATH = "detection_configs/default.yaml"
    WARNING_TARGET_LEVEL = 5
    WARNING_CENTER_FRAME_OFFSET = (0.0, 0.0, 0.0)


_VALID_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")


# =========================
# Code-level defaults (edit these if needed)
# =========================
DEFAULT_DATASET_ROOT = (
    "/home/femi/yolo_pose_dataset_creation/"
    "aircraft_pose_with_normalising_applied_multifield_only_3_2"
)
DEFAULT_H5_SOURCE = "/home/femi/Benchmarking_framework/Data/warning_b_test_h5"
DEFAULT_SPLIT = "val"
DEFAULT_SAMPLE_STEM = ""
DEFAULT_MAX_SAMPLES = 2000
DEFAULT_SHUFFLE = True
DEFAULT_SEED = 123
DEFAULT_LABEL_INDEX = 0
DEFAULT_KP_PATCH_RADIUS = 3
DEFAULT_YAML_KP_NAMES = (
    "/home/femi/yolo_pose_dataset_creation/"
    "aircraft_pose_with_normalising_applied_multifield_only_3_2/aircraft_pose.yaml"
)
DEFAULT_INCLUDE_HIDDEN = False
DEFAULT_APPLY_EXPORT_ROLL = True
DEFAULT_MAX_POINTS = 250000
DEFAULT_SPHERE_RADIUS = 0.45

# Code-level toggles (edit in code)
VISUALIZATION_ENABLED: bool = True
VISUALIZE_ONLY_WARNING_FAIL_CASES: bool = False
WARNING_OVERALL_PASS_FAIL_SUMMARY_ENABLED: bool = True
WARNING_PASS_FAIL_CSV_ENABLED: bool = True
WARNING_PASS_FAIL_CSV_PATH: str = ""
OPEN_KEYPOINT_EDITOR_ON_WARNING_FAIL: bool = False
RECHECK_WARNING_AFTER_FAIL_EDIT: bool = True
FAIL_EDITOR_PICK_RADIUS: float = 12.0
REOPEN_EDITOR_IF_RECHECK_FAIL: bool = True
MAX_FAIL_EDIT_ROUNDS: int = 10
A380_USE_DERIVED_NOSE_GEAR_LOGIC: bool = True
A380_DERIVED_NOSE_GEAR_BOX_EXTENT_M: Tuple[float, float, float] = (2.0, 2.0, 2.0)
WARNING_USE_NEAR_MARGIN_PASS: bool = True
WARNING_NEAR_MARGIN_M: float = 1
SKIP_FRONT_GEAR_CHECK_FOR_777_300ER: bool = False

DEFAULT_WARNING_CHECK = True
DEFAULT_WARNING_PASS_FAIL = True
DEFAULT_WARNING_PROFILE_CSV = str(WARNING_PROFILE_CSV)
DEFAULT_WARNING_YAML_COLUMN = str(WARNING_YAML_COLUMN)
DEFAULT_WARNING_YAML_ROOT = str(WARNING_YAML_ROOT)
DEFAULT_WARNING_YAML_RELPATH = str(WARNING_YAML_RELPATH)
DEFAULT_WARNING_TARGET_LEVEL = int(WARNING_TARGET_LEVEL)
DEFAULT_WARNING_BOX_SCALE = 1.0
DEFAULT_WARNING_H5_ROOT = ""
DEFAULT_WARNING_SCENE_TRANSFORM = True
DEFAULT_WARNING_FALLBACK_KP_NAMES = "front_wheels_mid,engine_left_box_center,engine_right_box_center"

# Visualization colors (fixed)
DEFAULT_BASE_CLOUD_GRAY_RGB = (0.72, 0.72, 0.72)
DEFAULT_KEYPOINT_BLUE_RGB = (0.10, 0.55, 1.00)


@dataclass
class SamplePaths:
    split: str
    stem: str
    image_path: Path
    label_path: Path


@dataclass
class YoloPoseLabel:
    class_id: int
    bbox_xyxy: Tuple[int, int, int, int]
    keypoints_px: np.ndarray  # (K,2), float
    keypoints_vis: np.ndarray  # (K,), float


@dataclass
class WarningPassFailSummary:
    evaluated: int = 0
    passed: int = 0
    failed: int = 0
    checks_total: int = 0
    checks_inside: int = 0
    checks_outside: int = 0
    fail_reasons: Dict[str, int] = field(default_factory=dict)


def _init_warning_runtime(
    warning_check_enabled: bool,
    *,
    warning_profile_csv: str,
    warning_yaml_column: str,
    warning_yaml_root: str,
    warning_yaml_relpath: str,
    warning_target_level: int,
    warning_h5_root: str,
    use_scene_h5_transform: bool,
) -> Tuple[Optional[Any], Dict[str, Any]]:
    if not warning_check_enabled:
        return None, {}

    try:
        import view_pcd_dir as pcd_view
    except Exception as e:
        raise RuntimeError(
            f"Failed to import warning-check module 'view_pcd_dir': {e}"
        ) from e

    warning_state: Dict[str, Any] = {
        "profile_map": pcd_view._load_warning_profile_map(str(warning_profile_csv or "")),
        "yaml_cache": {},
        "h5_path_cache": {},
        "scene_keypoints_cache": {},
        "warning_yaml_column": str(warning_yaml_column),
        "warning_yaml_root": str(warning_yaml_root),
        "warning_yaml_relpath": str(warning_yaml_relpath),
        "warning_target_level": int(warning_target_level),
        "warning_center_frame_offset": tuple(WARNING_CENTER_FRAME_OFFSET),
        "warning_h5_root": str(warning_h5_root),
        "use_scene_h5_transform": bool(use_scene_h5_transform),
    }
    return pcd_view, warning_state


def _write_warning_pass_fail_csv(
    out_csv: Path,
    rows: List[Dict[str, Any]],
    summary: WarningPassFailSummary,
) -> None:
    fieldnames = [
        "row_type",
        "split",
        "stem",
        "h5_stem",
        "scene_name",
        "h5_path",
        "status",
        "reason",
        "checks_total",
        "checks_inside",
        "checks_outside",
        "evaluated",
        "passed",
        "failed",
        "count",
    ]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})

        w.writerow(
            {
                "row_type": "summary",
                "status": "ALL",
                "checks_total": int(summary.checks_total),
                "checks_inside": int(summary.checks_inside),
                "checks_outside": int(summary.checks_outside),
                "evaluated": int(summary.evaluated),
                "passed": int(summary.passed),
                "failed": int(summary.failed),
            }
        )

        for reason, count in sorted(summary.fail_reasons.items(), key=lambda kv: (-kv[1], kv[0])):
            w.writerow(
                {
                    "row_type": "fail_reason",
                    "status": "FAIL",
                    "reason": str(reason),
                    "count": int(count),
                }
            )


def _open_keypoint_editor_for_fail(
    sample: SamplePaths,
    *,
    yaml_kp_names: str,
    label_index: int,
    pick_radius: float,
) -> Optional[str]:
    try:
        import adjust_keypoint_pixels as kp_editor
    except Exception as e:
        print(f"[warn] Failed to import adjust_keypoint_pixels: {e}")
        return None

    yaml_path: Optional[Path] = None
    yaml_raw = str(yaml_kp_names or "").strip()
    if yaml_raw:
        cand = Path(yaml_raw).expanduser()
        if cand.exists() and cand.is_file():
            yaml_path = cand
        else:
            print(f"[warn] yaml-kp-names not found, opening editor without YAML names: {cand}")

    try:
        action = kp_editor.run(
            image_path=sample.image_path,
            label_path=sample.label_path,
            yaml_path=yaml_path,
            out_label=sample.label_path,
            line_index=int(label_index),
            pick_radius=float(pick_radius),
            window_name=f"Adjust keypoints: {sample.stem}",
        )
        return str(action)
    except Exception as e:
        print(f"[warn] Keypoint editor failed for {sample.stem}: {e}")
        return None


def _warning_report_is_fail(report: Optional[Dict[str, Any]]) -> bool:
    return bool(report) and bool(report.get("evaluated", False)) and str(report.get("label", "FAIL")) == "FAIL"


def _editor_action_requests_skip(action: Optional[str]) -> bool:
    a = str(action or "").strip().lower()
    return a in {"quit", "skip", "cancel", "abort"}


def _normalize_key(name: str) -> str:
    s = str(name).strip().lower()
    for ch in ("-", " ", "\t"):
        s = s.replace(ch, "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def _is_a380_scene(stem: str) -> bool:
    try:
        h5_stem, _ = _parse_unique_scene_stem(stem)
    except Exception:
        h5_stem = str(stem)
    return "a380_800" in str(h5_stem).lower()


def _is_777_300er_scene(stem: str) -> bool:
    try:
        h5_stem, _ = _parse_unique_scene_stem(stem)
    except Exception:
        h5_stem = str(stem)
    return "777_300er" in str(h5_stem).lower()


def _find_scene_keypoint_by_alias(
    names: Sequence[str],
    xyz: np.ndarray,
    aliases: Sequence[str],
) -> Optional[np.ndarray]:
    if len(names) != int(xyz.shape[0]):
        return None

    norm_names = [_normalize_key(n) for n in names]
    by_norm: Dict[str, int] = {n: i for i, n in enumerate(norm_names)}

    for al in aliases:
        k = _normalize_key(al)
        if k in by_norm:
            return np.asarray(xyz[by_norm[k]], dtype=np.float64).reshape(3)

    for al in aliases:
        k = _normalize_key(al)
        for i, nm in enumerate(norm_names):
            if k and k in nm:
                return np.asarray(xyz[i], dtype=np.float64).reshape(3)
    return None


def _warning_eval_point_against_spec(
    point_world: np.ndarray,
    spec: Dict[str, Any],
    pcd_view_module: Any,
) -> Tuple[bool, np.ndarray, np.ndarray, float]:
    inside, local = pcd_view_module._point_inside_warning_box(
        np.asarray(point_world, dtype=np.float64),
        spec,
    )
    local = np.asarray(local, dtype=np.float64).reshape(3)
    half = np.asarray(spec["half"], dtype=np.float64).reshape(3)
    outside = np.maximum(np.abs(local) - half, 0.0)
    dist = float(np.linalg.norm(outside))
    return bool(inside), local, half, dist


def _build_derived_nose_spec_from_scene(
    *,
    sample_stem: str,
    pcd_view_module: Optional[Any],
    warning_state: Dict[str, Any],
    extent_m: Tuple[float, float, float],
) -> Tuple[Optional[Dict[str, Any]], str]:
    if pcd_view_module is None:
        return None, "warning module unavailable"

    names_scene, xyz_scene, scene_reason = pcd_view_module._load_scene_keypoints_from_h5(
        unique_scene=str(sample_stem),
        warning_state=warning_state,
    )
    if names_scene is None or xyz_scene is None:
        return None, str(scene_reason or "scene keypoints unavailable")

    p_fl = _find_scene_keypoint_by_alias(
        names_scene,
        xyz_scene,
        aliases=(
            "plane_front_left_wheel_link",
            "front_left_wheel_link",
            "left_front_wheel_link",
        ),
    )
    p_fr = _find_scene_keypoint_by_alias(
        names_scene,
        xyz_scene,
        aliases=(
            "plane_front_right_wheel_link",
            "front_right_wheel_link",
            "right_front_wheel_link",
        ),
    )
    if p_fl is None or p_fr is None:
        return None, "front wheel links missing in scene keypoints"

    center = 0.5 * (np.asarray(p_fl, dtype=np.float64) + np.asarray(p_fr, dtype=np.float64))
    extent = np.asarray(extent_m, dtype=np.float64).reshape(3)
    half = np.maximum(1e-3, 0.5 * np.abs(extent))
    spec = {
        "source_name": "derived_nose_gear",
        "center_world": center.reshape(3),
        "half": half.reshape(3),
        "R_world_to_box": np.eye(3, dtype=np.float64),
    }
    return spec, ""


def _is_front_like_kp_name(name: str) -> bool:
    n = _normalize_key(name)
    return ("front" in n) or ("nose" in n)


def _ensure_front_warning_check_present(
    *,
    sample_stem: str,
    kp_named: List[Tuple[str, np.ndarray]],
    warning_specs: Dict[str, Dict[str, Any]],
    warning_checks: List[Dict[str, Any]],
    pcd_view_module: Optional[Any],
    warning_state: Dict[str, Any],
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]], str]:
    front_kps = [
        (str(nm), np.asarray(p, dtype=np.float64).reshape(3))
        for nm, p in kp_named
        if _is_front_like_kp_name(str(nm))
    ]
    if not front_kps:
        return warning_specs, warning_checks, ""

    checked_front_names = {
        str(c.get("kp_name", ""))
        for c in warning_checks
        if str(c.get("box_key", "")) in {"front_landing_gear", "derived_nose_gear"}
    }
    missing_front = [(nm, p) for nm, p in front_kps if nm not in checked_front_names]
    if not missing_front:
        return warning_specs, warning_checks, ""

    specs_out = dict(warning_specs)
    front_spec = specs_out.get("front_landing_gear")
    derived_spec = specs_out.get("derived_nose_gear")
    if derived_spec is None:
        derived_spec, why = _build_derived_nose_spec_from_scene(
            sample_stem=str(sample_stem),
            pcd_view_module=pcd_view_module,
            warning_state=warning_state,
            extent_m=tuple(A380_DERIVED_NOSE_GEAR_BOX_EXTENT_M),
        )
        if derived_spec is not None:
            specs_out["derived_nose_gear"] = derived_spec
        elif front_spec is None:
            return warning_specs, warning_checks, f"front-check-missing: {why}"

    checks_out: List[Dict[str, Any]] = list(warning_checks)
    added = 0
    for nm, p in missing_front:
        candidates: List[Tuple[str, Dict[str, Any], bool, np.ndarray, np.ndarray, float]] = []
        if front_spec is not None:
            inside_f, local_f, half_f, dist_f = _warning_eval_point_against_spec(
                p,
                front_spec,
                pcd_view_module,
            )
            candidates.append(("front_landing_gear", front_spec, inside_f, local_f, half_f, dist_f))
        if derived_spec is not None:
            inside_d, local_d, half_d, dist_d = _warning_eval_point_against_spec(
                p,
                derived_spec,
                pcd_view_module,
            )
            candidates.append(("derived_nose_gear", derived_spec, inside_d, local_d, half_d, dist_d))
        if not candidates:
            continue

        best = min(candidates, key=lambda x: float(x[5]))
        box_key, _, inside, local, half, _ = best
        checks_out.append(
            {
                "kp_name": str(nm),
                "box_key": str(box_key),
                "inside": bool(inside),
                "abs_local": np.abs(np.asarray(local, dtype=np.float64).reshape(3)),
                "half": np.asarray(half, dtype=np.float64).reshape(3),
                "point": np.asarray(p, dtype=np.float64).reshape(3),
                "engine_lr_swapped": False,
            }
        )
        added += 1

    note = ""
    if added > 0:
        note = f"front-check-added={int(added)}"
    return specs_out, checks_out, note


def _apply_a380_derived_nose_logic(
    *,
    sample_stem: str,
    kp_named: List[Tuple[str, np.ndarray]],
    warning_specs: Dict[str, Dict[str, Any]],
    warning_checks: List[Dict[str, Any]],
    pcd_view_module: Optional[Any],
    warning_state: Dict[str, Any],
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]], str]:
    if not bool(A380_USE_DERIVED_NOSE_GEAR_LOGIC):
        return warning_specs, warning_checks, ""
    if not _is_a380_scene(str(sample_stem)):
        return warning_specs, warning_checks, ""
    if pcd_view_module is None:
        return warning_specs, warning_checks, "a380-derived-nose skipped: warning module unavailable"

    front_spec = warning_specs.get("front_landing_gear")
    if front_spec is None:
        return warning_specs, warning_checks, "a380-derived-nose skipped: no front_landing_gear spec"

    derived_spec, why = _build_derived_nose_spec_from_scene(
        sample_stem=str(sample_stem),
        pcd_view_module=pcd_view_module,
        warning_state=warning_state,
        extent_m=tuple(A380_DERIVED_NOSE_GEAR_BOX_EXTENT_M),
    )
    if derived_spec is None:
        return warning_specs, warning_checks, f"a380-derived-nose skipped: {why}"

    specs_out = dict(warning_specs)
    specs_out["derived_nose_gear"] = derived_spec

    if not warning_checks:
        return specs_out, warning_checks, "a380-derived-nose active (no front checks to remap)"

    kp_by_name: Dict[str, np.ndarray] = {
        str(nm): np.asarray(p, dtype=np.float64).reshape(3)
        for nm, p in kp_named
    }
    checks_out: List[Dict[str, Any]] = []
    remapped = 0

    for ck in warning_checks:
        c = dict(ck)
        box_key = str(c.get("box_key", ""))
        if box_key != "front_landing_gear":
            checks_out.append(c)
            continue

        kp_name = str(c.get("kp_name", ""))
        p = kp_by_name.get(kp_name)
        if p is None:
            checks_out.append(c)
            continue

        inside_front, local_front, half_front, dist_front = _warning_eval_point_against_spec(
            p,
            front_spec,
            pcd_view_module,
        )
        inside_derived, local_derived, half_derived, dist_derived = _warning_eval_point_against_spec(
            p,
            derived_spec,
            pcd_view_module,
        )

        if dist_derived < dist_front:
            c["box_key"] = "derived_nose_gear"
            c["inside"] = bool(inside_derived)
            c["abs_local"] = np.abs(local_derived)
            c["half"] = half_derived
            c["point"] = p
            remapped += 1
        else:
            c["inside"] = bool(inside_front)
            c["abs_local"] = np.abs(local_front)
            c["half"] = half_front
            c["point"] = p
        checks_out.append(c)

    note = "a380-derived-nose active"
    if remapped > 0:
        note += f" (remapped_front_checks={remapped})"
    return specs_out, checks_out, note


def _skip_front_checks_for_777_300er(
    *,
    sample_stem: str,
    warning_checks: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int, str]:
    if not bool(SKIP_FRONT_GEAR_CHECK_FOR_777_300ER):
        return warning_checks, 0, ""
    if not _is_777_300er_scene(str(sample_stem)):
        return warning_checks, 0, ""
    if not warning_checks:
        return warning_checks, 0, ""

    kept: List[Dict[str, Any]] = []
    removed = 0
    for ck in warning_checks:
        box_key = str(ck.get("box_key", ""))
        if box_key in {"front_landing_gear", "derived_nose_gear"}:
            removed += 1
            continue
        kept.append(ck)

    note = ""
    if removed > 0:
        note = f"777-front-check-skipped={int(removed)}"
    return kept, int(removed), note


def _outside_distance_from_warning_check(check: Dict[str, Any]) -> Optional[float]:
    try:
        abs_local = np.asarray(check.get("abs_local", None), dtype=np.float64).reshape(-1)
        half = np.asarray(check.get("half", None), dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if abs_local.size != 3 or half.size != 3:
        return None
    if not np.all(np.isfinite(abs_local)) or not np.all(np.isfinite(half)):
        return None
    outside = np.maximum(abs_local - half, 0.0)
    return float(np.linalg.norm(outside))


def _apply_warning_near_margin_pass(
    checks: List[Dict[str, Any]],
    *,
    near_margin_m: float,
) -> Tuple[List[Dict[str, Any]], int]:
    if not checks:
        return checks, 0
    margin = max(0.0, float(near_margin_m))
    if margin <= 0.0:
        return checks, 0

    out: List[Dict[str, Any]] = []
    promoted = 0
    for ck in checks:
        c = dict(ck)
        inside = bool(c.get("inside", False))
        c["inside_raw"] = bool(inside)
        c["near_pass"] = False

        if not inside:
            d = _outside_distance_from_warning_check(c)
            if d is not None:
                c["outside_distance_m"] = float(d)
                if float(d) <= margin:
                    c["inside"] = True
                    c["near_pass"] = True
                    promoted += 1

        out.append(c)
    return out, int(promoted)


def _decode_columns(cols_raw) -> List[str]:
    out: List[str] = []
    if cols_raw is None:
        return out
    for c in cols_raw:
        if isinstance(c, (bytes, bytearray)):
            out.append(c.decode("utf-8"))
        else:
            out.append(str(c))
    return out


def _list_h5_paths_any(source: str) -> List[str]:
    src = str(source).strip()
    if src.startswith("gs://"):
        try:
            from io_helpers import list_h5_paths as _list_h5_paths
        except Exception as e:
            raise RuntimeError(
                f"io_helpers is required for gs:// sources and failed to import: {e}"
            ) from e
        return _list_h5_paths(src)

    p = Path(src).expanduser().resolve()
    if not p.exists() or not p.is_dir():
        raise RuntimeError(f"h5-source is not a directory: {p}")
    out = sorted([str(x) for x in p.rglob("*.h5")] + [str(x) for x in p.rglob("*.H5")])
    return out


def _open_h5_any(path: str):
    p = str(path).strip()
    if p.startswith("gs://"):
        try:
            from io_helpers import open_h5_any as _open_h5
        except Exception as e:
            raise RuntimeError(
                f"io_helpers is required for gs:// sources and failed to import: {e}"
            ) from e
        return _open_h5(p)

    try:
        import h5py
    except Exception as e:
        raise RuntimeError(f"h5py is required to read local H5 files: {e}") from e
    return h5py.File(p, "r")


def _parse_unique_scene_stem(stem: str) -> Tuple[str, str]:
    s = str(stem).strip()
    if "__" not in s:
        raise ValueError(
            f"Stem '{s}' does not contain '__'. Expected '<h5_stem>__<scene_name>'."
        )
    h5_stem, scene_name = s.rsplit("__", 1)
    if not h5_stem or not scene_name:
        raise ValueError(f"Could not parse h5 stem/scene from '{s}'.")
    return h5_stem, scene_name


def _load_keypoint_names_from_yaml(yaml_path: Optional[str]) -> List[str]:
    if not yaml_path:
        return []
    p = Path(yaml_path).expanduser()
    if not p.exists() or not p.is_file():
        return []

    names: List[str] = []
    in_block = False
    for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines():
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


def _bbox_from_mask(mask2d: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.nonzero(mask2d)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _find_best_azimuth_roll(mask2d: np.ndarray) -> int:
    _, w = mask2d.shape
    if w <= 1:
        return 0

    col_has = mask2d.any(axis=0)
    empty = ~col_has
    if not np.any(empty):
        return 0

    best_len = 0
    best_start = 0
    for start in range(w):
        if not empty[start]:
            continue
        length = 0
        while length < w and empty[(start + length) % w]:
            length += 1
        if length > best_len:
            best_len = length
            best_start = start

    if best_len <= 0:
        return 0

    seam_col = (best_start + best_len // 2) % w
    shift = -int(seam_col)
    if shift % w == 0:
        return 0
    return shift


def _wrap_aware_bbox(mask2d: np.ndarray) -> Tuple[Optional[Tuple[int, int, int, int]], int]:
    bb = _bbox_from_mask(mask2d)
    if bb is None:
        return None, 0

    w = mask2d.shape[1]
    if w <= 1:
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


def _to_signed_roll(shift: int, w: int) -> int:
    if w <= 1:
        return 0
    s = int(shift) % int(w)
    if s > (w // 2):
        s -= int(w)
    return int(s)


def _compute_export_like_roll(mask2d: Optional[np.ndarray]) -> int:
    if mask2d is None:
        return 0

    h, w = mask2d.shape
    if h <= 0 or w <= 1 or not bool(ROLL_WIDE_BBOX):
        return 0

    bb = _bbox_from_mask(mask2d)
    if bb is None:
        return 0

    x1, _, x2, _ = bb
    bbox_frac = (x2 - x1 + 1) / float(w)
    shift_total = 0
    m = mask2d

    bb_wrap, shift_wrap = _wrap_aware_bbox(m)
    if bb_wrap is not None and shift_wrap != 0:
        m = np.roll(m, shift=int(shift_wrap), axis=1)
        shift_total += int(shift_wrap)
        x1w, _, x2w, _ = bb_wrap
        bbox_frac = (x2w - x1w + 1) / float(w)

    if bbox_frac > float(ROLL_WIDE_BBOX_FRAC):
        shift = _find_best_azimuth_roll(m)
        if shift == 0:
            shift = int(ROLL_WIDE_BBOX_COLS) % int(w)
        if shift != 0:
            shift_total += int(shift)

    return _to_signed_roll(shift_total, w)


def _extract_is_aircraft_mask(flat: np.ndarray, cols: Sequence[str], h: int, w: int) -> Optional[np.ndarray]:
    col_to_idx = {str(c).strip().lower(): i for i, c in enumerate(cols)}
    idx = None
    for key in ("is_aircraft", "aircraft_mask", "mask_aircraft", "aircraft"):
        if key in col_to_idx:
            idx = col_to_idx[key]
            break
    if idx is None:
        return None

    raw = flat[:, idx].reshape(h, w)
    if raw.dtype == np.bool_:
        return raw.copy()
    return np.asarray(raw > 0.5, dtype=bool)


def _load_scene_xyz_from_h5(
    h5_path: str,
    scene_name: str,
    apply_export_roll: bool,
) -> Tuple[np.ndarray, int]:
    with _open_h5_any(h5_path) as f:
        if scene_name not in f:
            raise RuntimeError(f"Scene '{scene_name}' not found in {Path(h5_path).name}")

        h = int(f.attrs["height"])
        w = int(f.attrs["width"])

        grp = f[scene_name]
        if "points" not in grp:
            raise RuntimeError(f"Scene '{scene_name}' has no points dataset")

        ds = grp["points"]
        flat = ds[()]
        cols = _decode_columns(ds.attrs.get("columns", None))
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

        idx = {name.strip().lower(): i for i, name in enumerate(cols)}
        for key in ("x", "y", "z"):
            if key not in idx:
                raise RuntimeError(f"Missing '{key}' column in H5 scene '{scene_name}'")

        xyz = np.stack(
            [
                flat[:, idx["x"]],
                flat[:, idx["y"]],
                flat[:, idx["z"]],
            ],
            axis=1,
        ).astype(np.float64)
        xyz_hw3 = xyz.reshape(h, w, 3)
        finite = np.all(np.isfinite(xyz_hw3), axis=2)
        xyz_hw3[~finite] = np.nan

        shift_cols = 0
        if apply_export_roll:
            mask_aircraft = _extract_is_aircraft_mask(flat, cols, h, w)
            shift_cols = _compute_export_like_roll(mask_aircraft)
            if shift_cols != 0:
                xyz_hw3 = np.roll(xyz_hw3, shift=shift_cols, axis=1)

    return xyz_hw3, int(shift_cols)


def _read_rgb_image(path: Path) -> np.ndarray:
    try:
        import cv2
    except Exception as e:
        raise RuntimeError(f"opencv-python is required to read dataset images: {e}") from e
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _clip_bbox(x1: int, y1: int, x2: int, y2: int, h: int, w: int) -> Tuple[int, int, int, int]:
    x1 = int(np.clip(x1, 0, w - 1))
    x2 = int(np.clip(x2, 0, w - 1))
    y1 = int(np.clip(y1, 0, h - 1))
    y2 = int(np.clip(y2, 0, h - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def _norm_to_px(v: float, size: int) -> float:
    if v <= 1.0:
        return float(v) * float(size)
    return float(v)


def _parse_yolo_pose_label_line(line: str, h: int, w: int) -> YoloPoseLabel:
    toks = [t for t in str(line).strip().split() if t]
    if len(toks) < 5:
        raise RuntimeError("Label line has fewer than 5 values")

    vals = [float(t) for t in toks]
    class_id = int(round(vals[0]))

    cx = _norm_to_px(vals[1], w)
    cy = _norm_to_px(vals[2], h)
    bw = _norm_to_px(vals[3], w)
    bh = _norm_to_px(vals[4], h)

    x1 = int(round(cx - bw * 0.5))
    y1 = int(round(cy - bh * 0.5))
    x2 = int(round(cx + bw * 0.5))
    y2 = int(round(cy + bh * 0.5))
    x1, y1, x2, y2 = _clip_bbox(x1, y1, x2, y2, h, w)

    rest = vals[5:]
    if not rest:
        kp_xy = np.zeros((0, 2), dtype=np.float64)
        kp_vis = np.zeros((0,), dtype=np.float64)
        return YoloPoseLabel(class_id=class_id, bbox_xyxy=(x1, y1, x2, y2), keypoints_px=kp_xy, keypoints_vis=kp_vis)

    if len(rest) % 3 == 0:
        arr = np.asarray(rest, dtype=np.float64).reshape(-1, 3)
        kx = np.array([_norm_to_px(v, w) for v in arr[:, 0]], dtype=np.float64)
        ky = np.array([_norm_to_px(v, h) for v in arr[:, 1]], dtype=np.float64)
        kv = arr[:, 2].astype(np.float64)
    elif len(rest) % 2 == 0:
        arr = np.asarray(rest, dtype=np.float64).reshape(-1, 2)
        kx = np.array([_norm_to_px(v, w) for v in arr[:, 0]], dtype=np.float64)
        ky = np.array([_norm_to_px(v, h) for v in arr[:, 1]], dtype=np.float64)
        kv = np.ones((arr.shape[0],), dtype=np.float64)
    else:
        raise RuntimeError(
            "Label keypoint payload is not divisible by 2 or 3. "
            f"Got {len(rest)} values after bbox."
        )

    kp_xy = np.stack([kx, ky], axis=1)
    kp_vis = kv
    return YoloPoseLabel(class_id=class_id, bbox_xyxy=(x1, y1, x2, y2), keypoints_px=kp_xy, keypoints_vis=kp_vis)


def _load_label_object(label_path: Path, h: int, w: int, label_index: int) -> YoloPoseLabel:
    lines = [ln.strip() for ln in label_path.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError(f"Empty label file: {label_path}")
    if label_index < 0 or label_index >= len(lines):
        raise RuntimeError(
            f"label-index={label_index} out of range for {label_path.name} (objects={len(lines)})"
        )
    return _parse_yolo_pose_label_line(lines[label_index], h=h, w=w)


def _sample_xyz_nearest(
    xyz_hw3: np.ndarray,
    r0: int,
    c0: int,
    radius: int,
) -> Optional[np.ndarray]:
    h, w, _ = xyz_hw3.shape
    r0 = int(np.clip(r0, 0, h - 1))
    c0 = int(np.clip(c0, 0, w - 1))

    p = xyz_hw3[r0, c0]
    if np.all(np.isfinite(p)):
        return p.astype(np.float64)

    rr_min = max(0, r0 - int(radius))
    rr_max = min(h, r0 + int(radius) + 1)
    cc_min = max(0, c0 - int(radius))
    cc_max = min(w, c0 + int(radius) + 1)

    sub = xyz_hw3[rr_min:rr_max, cc_min:cc_max, :]
    valid = np.all(np.isfinite(sub), axis=2)
    if not np.any(valid):
        return None

    rr, cc = np.where(valid)
    rr_abs = rr + rr_min
    cc_abs = cc + cc_min
    d2 = (rr_abs.astype(np.float64) - float(r0)) ** 2 + (cc_abs.astype(np.float64) - float(c0)) ** 2
    i = int(np.argmin(d2))
    return xyz_hw3[rr_abs[i], cc_abs[i], :].astype(np.float64)


def _pair_samples_for_split(dataset_root: Path, split: str) -> List[SamplePaths]:
    labels_dir = dataset_root / "labels" / split
    images_dir = dataset_root / "images" / split

    if not labels_dir.exists() or not labels_dir.is_dir():
        raise RuntimeError(f"Missing labels directory: {labels_dir}")
    if not images_dir.exists() or not images_dir.is_dir():
        raise RuntimeError(f"Missing images directory: {images_dir}")

    out: List[SamplePaths] = []
    for label_path in sorted(labels_dir.glob("*.txt")):
        stem = label_path.stem
        image_path = None
        for ext in _VALID_IMAGE_EXTS:
            cand = images_dir / f"{stem}{ext}"
            if cand.exists() and cand.is_file():
                image_path = cand
                break
        if image_path is None:
            continue
        out.append(
            SamplePaths(
                split=split,
                stem=stem,
                image_path=image_path,
                label_path=label_path,
            )
        )
    return out


def _collect_samples(dataset_root: Path, split: str, sample_stem: str) -> List[SamplePaths]:
    splits = ["train", "val", "test"] if split == "all" else [split]

    all_samples: List[SamplePaths] = []
    for sp in splits:
        all_samples.extend(_pair_samples_for_split(dataset_root, sp))

    if sample_stem:
        all_samples = [s for s in all_samples if s.stem == sample_stem]
        if not all_samples:
            raise RuntimeError(f"No sample found with stem='{sample_stem}' in split='{split}'")

    if not all_samples:
        raise RuntimeError("No image/label pairs found.")

    return all_samples


def _build_h5_stem_index(h5_source: str) -> Dict[str, List[str]]:
    h5_paths = _list_h5_paths_any(h5_source)
    if not h5_paths:
        raise RuntimeError(f"No .h5 files found under: {h5_source}")

    by_stem: Dict[str, List[str]] = {}
    for hp in h5_paths:
        stem = Path(hp).stem
        by_stem.setdefault(stem, []).append(hp)
    return by_stem


def _build_open3d_geometries(
    xyz_hw3: np.ndarray,
    rgb_hw3: np.ndarray,
    bbox: Tuple[int, int, int, int],
    kp_xyz: List[Tuple[str, np.ndarray]],
    max_points: int,
    sphere_radius: float,
    warning_specs: Optional[Dict[str, Dict[str, Any]]] = None,
    warning_checks: Optional[List[Dict[str, Any]]] = None,
    pcd_view_module: Optional[Any] = None,
) -> List[object]:
    try:
        import open3d as o3d
    except Exception as e:
        raise RuntimeError(
            f"Open3D is not installed or failed to import: {e}. Install with 'pip install open3d'."
        )

    geoms: List[object] = []
    _ = rgb_hw3  # kept for interface compatibility; scene cloud is rendered in fixed gray.

    valid = np.all(np.isfinite(xyz_hw3), axis=2)
    pts = xyz_hw3[valid].reshape(-1, 3).astype(np.float64)
    base_rgb = np.asarray(DEFAULT_BASE_CLOUD_GRAY_RGB, dtype=np.float64).reshape(1, 3)
    cols = np.tile(base_rgb, (pts.shape[0], 1))

    if max_points > 0 and pts.shape[0] > max_points:
        idx = np.random.choice(pts.shape[0], size=int(max_points), replace=False)
        pts = pts[idx]
        cols = cols[idx]

    base = o3d.geometry.PointCloud()
    base.points = o3d.utility.Vector3dVector(pts)
    base.colors = o3d.utility.Vector3dVector(cols)
    geoms.append(base)

    x1, y1, x2, y2 = bbox
    sub = xyz_hw3[y1 : y2 + 1, x1 : x2 + 1, :].reshape(-1, 3)
    sub = sub[np.all(np.isfinite(sub), axis=1)]
    if sub.shape[0] > 0:
        if max_points > 0 and sub.shape[0] > max_points // 2:
            idx2 = np.random.choice(sub.shape[0], size=max(1, max_points // 2), replace=False)
            sub = sub[idx2]
        bbox_cloud = o3d.geometry.PointCloud()
        bbox_cloud.points = o3d.utility.Vector3dVector(sub.astype(np.float64))
        bbox_cloud.paint_uniform_color([1.0, 0.12, 0.12])
        geoms.append(bbox_cloud)

    if warning_specs and pcd_view_module is not None:
        try:
            edges = list(pcd_view_module._warning_edges())
            edges_arr = np.asarray(edges, dtype=np.int32)
            for box_key, spec in sorted(warning_specs.items()):
                corners = pcd_view_module._oriented_box_corners(
                    center_world=np.asarray(spec["center_world"], dtype=np.float64),
                    half=np.asarray(spec["half"], dtype=np.float64),
                    R_world_to_box=np.asarray(spec["R_world_to_box"], dtype=np.float64),
                )
                ls = o3d.geometry.LineSet()
                ls.points = o3d.utility.Vector3dVector(corners.astype(np.float64))
                ls.lines = o3d.utility.Vector2iVector(edges_arr)
                col = np.asarray(pcd_view_module._warning_color_rgb01(str(box_key)), dtype=np.float64).reshape(1, 3)
                ls.colors = o3d.utility.Vector3dVector(np.tile(col, (edges_arr.shape[0], 1)))
                geoms.append(ls)
        except Exception as e:
            print(f"[warn] Failed to render warning boxes: {e}")

    kp_one_color = np.asarray(DEFAULT_KEYPOINT_BLUE_RGB, dtype=np.float64).reshape(3)
    check_by_name: Dict[str, bool] = {}
    if warning_checks:
        for c in warning_checks:
            nm = str(c.get("kp_name", ""))
            if nm:
                check_by_name[nm] = bool(c.get("inside", False))

    for kp_name, p3 in kp_xyz:
        sph = o3d.geometry.TriangleMesh.create_sphere(radius=float(sphere_radius), resolution=14)
        sph.compute_vertex_normals()
        if kp_name in check_by_name:
            if check_by_name[kp_name]:
                sph.paint_uniform_color([0.1, 0.85, 0.1])
            else:
                sph.paint_uniform_color([1.0, 0.15, 0.15])
        else:
            sph.paint_uniform_color(kp_one_color.tolist())
        sph.translate(np.asarray(p3, dtype=np.float64).reshape(3), relative=False)
        geoms.append(sph)

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
    geoms.append(frame)
    return geoms


def _run_view_for_sample(
    sample: SamplePaths,
    h5_path: str,
    scene_name: str,
    keypoint_names: List[str],
    label_index: int,
    kp_patch_radius: int,
    apply_export_roll: bool,
    include_hidden: bool,
    max_points: int,
    sphere_radius: float,
    visualize: bool,
    visualize_only_warning_fail_cases: bool,
    warning_check_enabled: bool,
    warning_pass_fail_enabled: bool,
    warning_box_scale: float,
    warning_fallback_kp_names: List[str],
    pcd_view_module: Optional[Any],
    warning_state: Dict[str, Any],
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    rgb = _read_rgb_image(sample.image_path)
    h, w = rgb.shape[:2]

    xyz_hw3, shift_cols = _load_scene_xyz_from_h5(
        h5_path=h5_path,
        scene_name=scene_name,
        apply_export_roll=apply_export_roll,
    )
    if xyz_hw3.shape[0] != h or xyz_hw3.shape[1] != w:
        raise RuntimeError(
            "Image and H5 grid size mismatch: "
            f"image={w}x{h}, h5_grid={xyz_hw3.shape[1]}x{xyz_hw3.shape[0]}"
        )

    label_obj = _load_label_object(sample.label_path, h=h, w=w, label_index=label_index)

    kp_xyz: List[Tuple[str, np.ndarray]] = []
    missing_kp = 0
    for i, (xy, vis) in enumerate(zip(label_obj.keypoints_px, label_obj.keypoints_vis)):
        if not include_hidden and float(vis) <= 0.0:
            continue

        ck = int(round(float(xy[0])))
        rk = int(round(float(xy[1])))
        p3 = _sample_xyz_nearest(
            xyz_hw3=xyz_hw3,
            r0=rk,
            c0=ck,
            radius=kp_patch_radius,
        )
        if p3 is None:
            missing_kp += 1
            continue

        if i < len(keypoint_names) and keypoint_names[i]:
            kp_name = keypoint_names[i]
        elif i < len(warning_fallback_kp_names):
            kp_name = str(warning_fallback_kp_names[i])
        else:
            kp_name = f"K{i}"
        kp_xyz.append((kp_name, p3))

    warning_report: Optional[Dict[str, Any]] = None
    warning_specs: Dict[str, Dict[str, Any]] = {}
    warning_checks: List[Dict[str, Any]] = []
    warning_reason = ""
    warning_mode_note = ""
    near_promoted_n = 0
    if warning_check_enabled:
        if not kp_xyz:
            warning_reason = "no 3D keypoints to check"
        elif pcd_view_module is None:
            warning_reason = "warning-check module unavailable"
        else:
            warning_specs, warning_checks, warning_reason = pcd_view_module._build_warning_specs_and_checks(
                unique_scene=str(sample.stem),
                kp_named=kp_xyz,
                warning_state=warning_state,
                warning_box_scale=float(warning_box_scale),
            )
            warning_specs, warning_checks, mode_note = _apply_a380_derived_nose_logic(
                sample_stem=str(sample.stem),
                kp_named=kp_xyz,
                warning_specs=warning_specs,
                warning_checks=warning_checks,
                pcd_view_module=pcd_view_module,
                warning_state=warning_state,
            )
            warning_mode_note = str(mode_note or "")
            warning_specs, warning_checks, front_note = _ensure_front_warning_check_present(
                sample_stem=str(sample.stem),
                kp_named=kp_xyz,
                warning_specs=warning_specs,
                warning_checks=warning_checks,
                pcd_view_module=pcd_view_module,
                warning_state=warning_state,
            )
            if front_note:
                warning_mode_note = f"{warning_mode_note}; {front_note}" if warning_mode_note else front_note
            warning_checks, _, skip_note = _skip_front_checks_for_777_300er(
                sample_stem=str(sample.stem),
                warning_checks=warning_checks,
            )
            if skip_note:
                warning_mode_note = f"{warning_mode_note}; {skip_note}" if warning_mode_note else skip_note
            if bool(WARNING_USE_NEAR_MARGIN_PASS):
                warning_checks, near_promoted_n = _apply_warning_near_margin_pass(
                    warning_checks,
                    near_margin_m=float(WARNING_NEAR_MARGIN_M),
                )

    if warning_check_enabled and warning_pass_fail_enabled:
        warning_report = {
            "evaluated": True,
            "label": "FAIL",
            "reason": "",
            "checks_total": 0,
            "checks_inside": 0,
            "checks_outside": 0,
        }
        if warning_checks:
            inside_n = sum(1 for c in warning_checks if bool(c.get("inside", False)))
            outside_n = int(len(warning_checks) - inside_n)
            warning_report["checks_total"] = int(len(warning_checks))
            warning_report["checks_inside"] = int(inside_n)
            warning_report["checks_outside"] = int(outside_n)
            if outside_n == 0:
                warning_report["label"] = "PASS"
            else:
                warning_report["reason"] = f"{outside_n} outside warning box"
        else:
            warning_report["reason"] = str(warning_reason or "no warning-box checks produced")

    do_visualize = bool(visualize)
    if do_visualize and bool(visualize_only_warning_fail_cases):
        if warning_report is not None and bool(warning_report.get("evaluated", False)):
            do_visualize = str(warning_report.get("label", "FAIL")) == "FAIL"
        else:
            do_visualize = False

    if do_visualize:
        geoms = _build_open3d_geometries(
            xyz_hw3=xyz_hw3,
            rgb_hw3=rgb,
            bbox=label_obj.bbox_xyxy,
            kp_xyz=kp_xyz,
            max_points=max_points,
            sphere_radius=sphere_radius,
            warning_specs=warning_specs,
            warning_checks=warning_checks,
            pcd_view_module=pcd_view_module,
        )

        try:
            import open3d as o3d
        except Exception as e:
            raise RuntimeError(f"Open3D unavailable: {e}")

        window_name = f"Dataset check: {sample.stem}"
        o3d.visualization.draw_geometries(geoms, window_name=window_name, width=1400, height=900)

    msg = (
        f"split={sample.split} stem={sample.stem} class={label_obj.class_id} "
        f"bbox={label_obj.bbox_xyxy} kps_3d={len(kp_xyz)}/{len(label_obj.keypoints_px)} "
        f"missing={missing_kp} roll={shift_cols}"
    )
    if warning_mode_note:
        msg += f" {warning_mode_note}"
    if near_promoted_n > 0:
        msg += (
            f" warning_near_pass={int(near_promoted_n)}"
            f"(<= {float(WARNING_NEAR_MARGIN_M):.2f}m)"
        )
    if warning_report is not None:
        wp_label = str(warning_report.get("label", "FAIL"))
        wp_reason = str(warning_report.get("reason", ""))
        wp_inside = int(warning_report.get("checks_inside", 0))
        wp_total = int(warning_report.get("checks_total", 0))
        if wp_reason:
            msg += f" warning={wp_label} ({wp_reason})"
        else:
            msg += f" warning={wp_label} inside={wp_inside}/{wp_total}"
    return True, msg, warning_report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Validate YOLO pose dataset by visualizing label keypoints in real H5 point cloud (Open3D)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset-root", type=str, default=DEFAULT_DATASET_ROOT, help="Dataset root with images/ and labels/")
    p.add_argument("--h5-source", type=str, default=DEFAULT_H5_SOURCE, help="Root folder (or gs://) containing original H5 files")
    p.add_argument(
        "--split",
        type=str,
        default=DEFAULT_SPLIT,
        choices=["train", "val", "test", "all"],
        help="Which split to view",
    )
    p.add_argument(
        "--sample-stem",
        type=str,
        default=DEFAULT_SAMPLE_STEM,
        help="Optional exact sample stem '<h5_stem>__<scene_name>'",
    )
    p.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES, help="How many samples to open (0 = all)")

    shuffle = p.add_mutually_exclusive_group()
    shuffle.add_argument("--shuffle", dest="shuffle", action="store_true", help="Shuffle sample order before viewing")
    shuffle.add_argument("--no-shuffle", dest="shuffle", action="store_false", help="Do not shuffle sample order")

    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Seed used when --shuffle is enabled")
    p.add_argument("--label-index", type=int, default=DEFAULT_LABEL_INDEX, help="Object line index in each label txt (default: first object)")
    p.add_argument("--kp-patch-radius", type=int, default=DEFAULT_KP_PATCH_RADIUS, help="Pixel radius to search nearest finite xyz for keypoints")
    p.add_argument("--yaml-kp-names", type=str, default=DEFAULT_YAML_KP_NAMES, help="Optional YOLO dataset yaml to name keypoints")

    hidden = p.add_mutually_exclusive_group()
    hidden.add_argument("--include-hidden", dest="include_hidden", action="store_true", help="Include keypoints with vis<=0")
    hidden.add_argument("--no-include-hidden", dest="include_hidden", action="store_false", help="Do not include keypoints with vis<=0")

    roll = p.add_mutually_exclusive_group()
    roll.add_argument(
        "--apply-export-roll",
        dest="apply_export_roll",
        action="store_true",
        help="Apply H5 azimuth roll re-alignment",
    )
    roll.add_argument(
        "--no-apply-export-roll",
        dest="apply_export_roll",
        action="store_false",
        help="Disable H5 azimuth roll re-alignment",
    )

    p.add_argument("--max-points", type=int, default=DEFAULT_MAX_POINTS, help="Max points for base cloud rendering (for speed)")
    p.add_argument("--sphere-radius", type=float, default=DEFAULT_SPHERE_RADIUS, help="Keypoint sphere radius in meters")

    warning_check = p.add_mutually_exclusive_group()
    warning_check.add_argument(
        "--warning-check",
        dest="warning_check",
        action="store_true",
        help="Enable warning-box inside/outside checks.",
    )
    warning_check.add_argument(
        "--no-warning-check",
        dest="warning_check",
        action="store_false",
        help="Disable warning-box inside/outside checks.",
    )

    warning_pf = p.add_mutually_exclusive_group()
    warning_pf.add_argument(
        "--warning-pass-fail",
        dest="warning_pass_fail",
        action="store_true",
        help="Enable per-sample warning PASS/FAIL evaluation.",
    )
    warning_pf.add_argument(
        "--no-warning-pass-fail",
        dest="warning_pass_fail",
        action="store_false",
        help="Disable per-sample warning PASS/FAIL evaluation.",
    )

    warning_scene_tf = p.add_mutually_exclusive_group()
    warning_scene_tf.add_argument(
        "--warning-scene-transform",
        dest="warning_scene_transform",
        action="store_true",
        help="Use source H5 scene keypoints to align warning boxes.",
    )
    warning_scene_tf.add_argument(
        "--no-warning-scene-transform",
        dest="warning_scene_transform",
        action="store_false",
        help="Disable source H5 scene keypoint alignment (fallback to observed keypoint fit).",
    )

    p.add_argument(
        "--warning-profile-csv",
        type=str,
        default=DEFAULT_WARNING_PROFILE_CSV,
        help="Profile CSV used to resolve recommended warning YAML per aircraft.",
    )
    p.add_argument(
        "--warning-yaml-column",
        type=str,
        default=DEFAULT_WARNING_YAML_COLUMN,
        help="Column name in --warning-profile-csv containing warning YAML path.",
    )
    p.add_argument(
        "--warning-yaml-root",
        type=str,
        default=DEFAULT_WARNING_YAML_ROOT,
        help="Fallback root for warning YAML lookup.",
    )
    p.add_argument(
        "--warning-yaml-relpath",
        type=str,
        default=DEFAULT_WARNING_YAML_RELPATH,
        help="Relative YAML path under each aircraft folder in --warning-yaml-root.",
    )
    p.add_argument(
        "--warning-target-level",
        type=int,
        default=DEFAULT_WARNING_TARGET_LEVEL,
        help="Preferred warning_level in YAML crop_boxes (fallbacks to all boxes if not found).",
    )
    p.add_argument(
        "--warning-box-scale",
        type=float,
        default=DEFAULT_WARNING_BOX_SCALE,
        help="Scale factor on warning-box half sizes for inside/outside checks.",
    )
    p.add_argument(
        "--warning-h5-root",
        type=str,
        default=DEFAULT_WARNING_H5_ROOT,
        help="Root folder containing source H5 files for warning scene transform (default: --h5-source).",
    )
    p.add_argument(
        "--warning-fallback-kp-names",
        type=str,
        default=DEFAULT_WARNING_FALLBACK_KP_NAMES,
        help=(
            "Comma-separated fallback keypoint names used when YAML names are missing "
            "(applied by keypoint index)."
        ),
    )

    p.set_defaults(
        shuffle=bool(DEFAULT_SHUFFLE),
        include_hidden=bool(DEFAULT_INCLUDE_HIDDEN),
        apply_export_roll=bool(DEFAULT_APPLY_EXPORT_ROLL),
        warning_check=bool(DEFAULT_WARNING_CHECK),
        warning_pass_fail=bool(DEFAULT_WARNING_PASS_FAIL),
        warning_scene_transform=bool(DEFAULT_WARNING_SCENE_TRANSFORM),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if not dataset_root.exists() or not dataset_root.is_dir():
        raise RuntimeError(f"dataset-root not found or not a directory: {dataset_root}")

    samples = _collect_samples(
        dataset_root=dataset_root,
        split=str(args.split),
        sample_stem=str(args.sample_stem).strip(),
    )

    if args.shuffle:
        rng = np.random.default_rng(int(args.seed))
        rng.shuffle(samples)

    if int(args.max_samples) > 0:
        samples = samples[: int(args.max_samples)]

    h5_by_stem = _build_h5_stem_index(str(args.h5_source))
    kp_names = _load_keypoint_names_from_yaml(str(args.yaml_kp_names).strip() or None)
    warning_h5_root = str(args.warning_h5_root).strip() or str(args.h5_source).strip()
    pcd_view_module, warning_state = _init_warning_runtime(
        bool(args.warning_check),
        warning_profile_csv=str(args.warning_profile_csv or ""),
        warning_yaml_column=str(args.warning_yaml_column or DEFAULT_WARNING_YAML_COLUMN),
        warning_yaml_root=str(args.warning_yaml_root or DEFAULT_WARNING_YAML_ROOT),
        warning_yaml_relpath=str(args.warning_yaml_relpath or DEFAULT_WARNING_YAML_RELPATH),
        warning_target_level=int(args.warning_target_level),
        warning_h5_root=str(warning_h5_root),
        use_scene_h5_transform=bool(args.warning_scene_transform),
    )
    warning_fallback_kp_names: List[str] = []
    if bool(args.warning_check) and pcd_view_module is not None:
        warning_fallback_kp_names = pcd_view_module._parse_name_csv(str(args.warning_fallback_kp_names))

    runtime_visualize = bool(VISUALIZATION_ENABLED)
    if runtime_visualize:
        try:
            import open3d as _o3d  # noqa: F401
        except Exception as e:
            runtime_visualize = False
            print(
                "[warn] Open3D unavailable; visualization disabled for this run. "
                f"Reason: {e}"
            )

    print(f"[info] dataset root: {dataset_root}")
    print(f"[info] h5 source: {args.h5_source}")
    print(f"[info] samples selected: {len(samples)}")
    print(
        "[info] code-toggle visualization: "
        f"{bool(VISUALIZATION_ENABLED)} (effective={bool(runtime_visualize)})"
    )
    print(
        "[info] code-toggle visualize only warning FAIL cases: "
        f"{bool(VISUALIZE_ONLY_WARNING_FAIL_CASES)}"
    )
    print(
        "[info] code-toggle open keypoint editor on warning FAIL: "
        f"{bool(OPEN_KEYPOINT_EDITOR_ON_WARNING_FAIL)}"
    )
    print(
        "[info] code-toggle recheck sample after fail edit: "
        f"{bool(RECHECK_WARNING_AFTER_FAIL_EDIT)}"
    )
    print(
        "[info] code-toggle reopen editor if recheck still fails: "
        f"{bool(REOPEN_EDITOR_IF_RECHECK_FAIL)}"
    )
    if bool(OPEN_KEYPOINT_EDITOR_ON_WARNING_FAIL):
        print("[info] editor skip key: press 'q' to skip this sample without recheck.")
    print(f"[info] code-toggle max fail edit rounds: {int(MAX_FAIL_EDIT_ROUNDS)}")
    print(
        "[info] code-toggle a380 derived nose-gear logic: "
        f"{bool(A380_USE_DERIVED_NOSE_GEAR_LOGIC)} "
        f"(extent_m={tuple(A380_DERIVED_NOSE_GEAR_BOX_EXTENT_M)})"
    )
    print(
        "[info] code-toggle warning near-margin pass: "
        f"{bool(WARNING_USE_NEAR_MARGIN_PASS)} "
        f"(margin_m={float(WARNING_NEAR_MARGIN_M):.2f})"
    )
    print(
        "[info] code-toggle skip front-gear check for 777_300er: "
        f"{bool(SKIP_FRONT_GEAR_CHECK_FOR_777_300ER)}"
    )
    if kp_names:
        print(f"[info] keypoint names loaded: {len(kp_names)}")
    print(f"[info] warning check: {'enabled' if bool(args.warning_check) else 'disabled'}")
    if bool(args.warning_check):
        print(f"[info] warning pass/fail: {'enabled' if bool(args.warning_pass_fail) else 'disabled'}")
        print(
            "[info] code-toggle overall warning pass/fail summary: "
            f"{bool(WARNING_OVERALL_PASS_FAIL_SUMMARY_ENABLED)}"
        )
        print(f"[info] warning h5 root: {warning_h5_root}")
        print(
            f"[info] warning scene transform: "
            f"{'enabled' if bool(args.warning_scene_transform) else 'disabled'}"
        )

    shown = 0
    skipped = 0
    warning_summary = WarningPassFailSummary()
    warning_csv_rows: List[Dict[str, Any]] = []
    warning_csv_path: Optional[Path] = None
    if bool(args.warning_check) and bool(args.warning_pass_fail) and bool(WARNING_PASS_FAIL_CSV_ENABLED):
        csv_path_raw = str(WARNING_PASS_FAIL_CSV_PATH or "").strip()
        if csv_path_raw:
            warning_csv_path = Path(csv_path_raw).expanduser().resolve()
        else:
            warning_csv_path = (dataset_root / "warning_pass_fail_results.csv").resolve()
        print(f"[info] code-toggle warning csv export: True ({warning_csv_path})")
    elif bool(args.warning_check) and bool(args.warning_pass_fail):
        print("[info] code-toggle warning csv export: False")

    for i, sample in enumerate(samples, 1):
        try:
            h5_stem, scene_name = _parse_unique_scene_stem(sample.stem)
        except Exception as e:
            skipped += 1
            print(f"[{i}/{len(samples)}][skip] bad stem {sample.stem}: {e}")
            continue

        matches = h5_by_stem.get(h5_stem, [])
        if not matches:
            skipped += 1
            print(f"[{i}/{len(samples)}][skip] no h5 for stem='{h5_stem}' sample={sample.stem}")
            continue
        if len(matches) > 1:
            print(f"[{i}/{len(samples)}][warn] multiple h5 matches for '{h5_stem}', using first: {Path(matches[0]).name}")

        h5_path = matches[0]

        try:
            ok, msg, warning_report = _run_view_for_sample(
                sample=sample,
                h5_path=h5_path,
                scene_name=scene_name,
                keypoint_names=kp_names,
                label_index=int(args.label_index),
                kp_patch_radius=int(args.kp_patch_radius),
                apply_export_roll=bool(args.apply_export_roll),
                include_hidden=bool(args.include_hidden),
                max_points=int(args.max_points),
                sphere_radius=float(args.sphere_radius),
                visualize=bool(runtime_visualize),
                visualize_only_warning_fail_cases=bool(VISUALIZE_ONLY_WARNING_FAIL_CASES),
                warning_check_enabled=bool(args.warning_check),
                warning_pass_fail_enabled=bool(args.warning_pass_fail),
                warning_box_scale=float(args.warning_box_scale),
                warning_fallback_kp_names=warning_fallback_kp_names,
                pcd_view_module=pcd_view_module,
                warning_state=warning_state,
            )
            if ok:
                shown += 1
                final_msg = str(msg)
                final_warning_report = warning_report
                if (
                    bool(OPEN_KEYPOINT_EDITOR_ON_WARNING_FAIL)
                    and bool(args.warning_check)
                    and bool(args.warning_pass_fail)
                    and _warning_report_is_fail(final_warning_report)
                ):
                    rounds_done = 0
                    while _warning_report_is_fail(final_warning_report):
                        if rounds_done >= int(MAX_FAIL_EDIT_ROUNDS):
                            print(
                                f"[{i}/{len(samples)}][edit] reached max fail edit rounds "
                                f"({int(MAX_FAIL_EDIT_ROUNDS)}) for {sample.stem}"
                            )
                            break

                        rounds_done += 1
                        print(
                            f"[{i}/{len(samples)}][edit] warning FAIL for {sample.stem}; "
                            f"opening keypoint editor (attempt {rounds_done}/{int(MAX_FAIL_EDIT_ROUNDS)})"
                        )
                        editor_action = _open_keypoint_editor_for_fail(
                            sample=sample,
                            yaml_kp_names=str(args.yaml_kp_names),
                            label_index=int(args.label_index),
                            pick_radius=float(FAIL_EDITOR_PICK_RADIUS),
                        )
                        if editor_action is None:
                            break

                        print(
                            f"[{i}/{len(samples)}][edit] editor action='{editor_action}' "
                            f"sample={sample.stem}"
                        )
                        if _editor_action_requests_skip(editor_action):
                            print(
                                f"[{i}/{len(samples)}][edit] skip requested "
                                f"(action='{editor_action}') sample={sample.stem}"
                            )
                            break

                        if not bool(RECHECK_WARNING_AFTER_FAIL_EDIT):
                            break

                        ok2, msg2, warning_report2 = _run_view_for_sample(
                            sample=sample,
                            h5_path=h5_path,
                            scene_name=scene_name,
                            keypoint_names=kp_names,
                            label_index=int(args.label_index),
                            kp_patch_radius=int(args.kp_patch_radius),
                            apply_export_roll=bool(args.apply_export_roll),
                            include_hidden=bool(args.include_hidden),
                            max_points=int(args.max_points),
                            sphere_radius=float(args.sphere_radius),
                            visualize=bool(runtime_visualize),
                            visualize_only_warning_fail_cases=bool(VISUALIZE_ONLY_WARNING_FAIL_CASES),
                            warning_check_enabled=bool(args.warning_check),
                            warning_pass_fail_enabled=bool(args.warning_pass_fail),
                            warning_box_scale=float(args.warning_box_scale),
                            warning_fallback_kp_names=warning_fallback_kp_names,
                            pcd_view_module=pcd_view_module,
                            warning_state=warning_state,
                        )
                        if ok2:
                            final_msg = str(msg2)
                            final_warning_report = warning_report2
                            print(f"[{i}/{len(samples)}][recheck] {final_msg}")

                        if not bool(REOPEN_EDITOR_IF_RECHECK_FAIL):
                            break

                print(f"[{i}/{len(samples)}][ok] {final_msg}")
                if final_warning_report and bool(final_warning_report.get("evaluated", False)):
                    warning_summary.evaluated += 1
                    warning_summary.checks_total += int(final_warning_report.get("checks_total", 0))
                    warning_summary.checks_inside += int(final_warning_report.get("checks_inside", 0))
                    warning_summary.checks_outside += int(final_warning_report.get("checks_outside", 0))
                    label = str(final_warning_report.get("label", "FAIL"))
                    if label == "PASS":
                        warning_summary.passed += 1
                    else:
                        warning_summary.failed += 1
                        reason = str(final_warning_report.get("reason", "") or "unknown warning failure")
                        warning_summary.fail_reasons[reason] = (
                            int(warning_summary.fail_reasons.get(reason, 0)) + 1
                        )
                    if warning_csv_path is not None:
                        warning_csv_rows.append(
                            {
                                "row_type": "sample",
                                "split": str(sample.split),
                                "stem": str(sample.stem),
                                "h5_stem": str(h5_stem),
                                "scene_name": str(scene_name),
                                "h5_path": str(h5_path),
                                "status": str(label),
                                "reason": str(final_warning_report.get("reason", "") or ""),
                                "checks_total": int(final_warning_report.get("checks_total", 0)),
                                "checks_inside": int(final_warning_report.get("checks_inside", 0)),
                                "checks_outside": int(final_warning_report.get("checks_outside", 0)),
                            }
                        )
        except Exception as e:
            skipped += 1
            print(f"[{i}/{len(samples)}][skip] {sample.stem}: {e}")

    print("\n[summary]")
    print(f"  shown: {shown}")
    print(f"  skipped: {skipped}")
    print(f"  total considered: {len(samples)}")
    if (
        bool(args.warning_check)
        and bool(args.warning_pass_fail)
        and bool(WARNING_OVERALL_PASS_FAIL_SUMMARY_ENABLED)
    ):
        print(
            f"  warning pass/fail: evaluated={warning_summary.evaluated} "
            f"pass={warning_summary.passed} fail={warning_summary.failed}"
        )
        print(
            f"  warning checks: total={warning_summary.checks_total} "
            f"inside={warning_summary.checks_inside} outside={warning_summary.checks_outside}"
        )
        if warning_summary.fail_reasons:
            for reason, count in sorted(
                warning_summary.fail_reasons.items(),
                key=lambda kv: (-kv[1], kv[0]),
            ):
                print(f"  warning fail-reason: {reason} -> {count}")
    if warning_csv_path is not None:
        _write_warning_pass_fail_csv(
            out_csv=warning_csv_path,
            rows=warning_csv_rows,
            summary=warning_summary,
        )
        print(f"  warning csv: {warning_csv_path} (rows={len(warning_csv_rows)})")


if __name__ == "__main__":
    main()
