#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Single-image pipeline:
1) Run YOLO pose on one test image (mapped to H5 scene by filename stem)
2) Backproject bbox/keypoints to PCD
3) Optionally visualize PCD
4) Optionally run warning-box inside/outside checks with PASS/FAIL summary

Expected image filename stem format:
  <h5_stem>__<scene_name>
Example:
  movement_737_900er__2025-09-11T19-56-15__scene_000.png
"""

from __future__ import annotations

import argparse
import csv
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
from io_helpers import list_h5_paths, open_h5_any


# =========================
# Code-level toggles (edit here)
# =========================
VISUALIZE_PCD: bool = True
RUN_WARNING_CHECK: bool = True
RUN_WARNING_PASS_FAIL: bool = True
USE_SCENE_H5_TRANSFORM: bool = True
SHOW_AXES: bool = False

# PCD/keypoint rendering/check settings
KPT_COUNT: int = 3
KPT_RADIUS_M: float = 0.25
WARNING_BOX_SCALE: float = 1.0
WARNING_FALLBACK_KP_NAMES: str = "front_wheels_mid,engine_left_box_center,engine_right_box_center"

# YOLO run settings
SAVE_DEBUG_IMAGE: bool = True
IMG_SIZE: int = 1024
YOLO_CONF: float = 0.05
DEVICE: str = "0"
KP_CONF_THR: float = 0.9
KP_PATCH_RADIUS: int = 3
CHECK_BBOX_COVERAGE: bool = True
BBOX_FULL_THR: float = 0.995
COMPARE_FAIL_KP_3D_ERROR: bool = True
EXPORT_GT_3D_FROM_2D: bool = True
USE_GT_3D_FROM_2D_FOR_FAIL_DISTANCE: bool = True
EXPORT_FAIL_DISTANCE_VIS: bool = True
FAIL_DISTANCE_TOP_N: int = 200
FAIL_DISTANCE_PLOT_MAX_KP: int = 20
FAIL_DISTANCE_PLOT_MAX_SCENE: int = 20
EXPORT_WARNING_BOX_DISTANCE: bool = True

# Default paths (can still be overridden by CLI args)
DEFAULT_IMAGE_PATH: str = "/home/femi/yolo_pose_dataset_creation/aircraft_pose_with_normalising_applied_multifield_only_3_2/images/val"
DEFAULT_WEIGHTS_PATH: str = "/home/femi/yolo_pose_dataset_creation/runs/pose/aircraft_pose_exp_after_point_clound_check/weights/best.pt"
DEFAULT_SOURCE_H5_ROOT: str = "/home/femi/Benchmarking_framework/Data/warning_b_test_h5"
DEFAULT_OUT_DIR: str = "/home/femi/yolo_pose_dataset_creation/pcd_from_yolo5"
DEFAULT_YAML_KP_NAMES: str = (
    "/home/femi/yolo_pose_dataset_creation/"
    "aircraft_pose_with_normalising_applied_multifield_only_3/aircraft_pose.yaml"
)


def _split_unique_scene(unique_scene: str) -> Tuple[str, Optional[str]]:
    s = str(unique_scene or "").strip()
    if "__" not in s:
        return s, None
    bag, scene = s.rsplit("__", 1)
    return bag, (scene if scene else None)


def _load_keypoint_names_from_yaml(yaml_path: Optional[str]) -> List[str]:
    if not yaml_path:
        return []
    p = Path(yaml_path).expanduser()
    if not p.exists() or not p.is_file():
        return []
    names: List[str] = []
    lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    in_block = False
    for ln in lines:
        s = str(ln).strip()
        if s == "keypoints:":
            in_block = True
            continue
        if in_block:
            if s.startswith("- "):
                names.append(str(s[2:]).strip())
            elif s and (not s.startswith("#")):
                break
    return names


def _infer_label_path_for_image(image_path: Path) -> Optional[Path]:
    ip = Path(image_path).expanduser().resolve()
    stem = str(ip.stem)
    candidates: List[Path] = []

    # Common YOLO layout: .../images/<split>/x.png -> .../labels/<split>/x.txt
    parts = list(ip.parts)
    if "images" in parts:
        idx = parts.index("images")
        repl = parts[:]
        repl[idx] = "labels"
        candidates.append(Path(*repl).with_suffix(".txt"))

    # Sibling label file.
    candidates.append(ip.with_suffix(".txt"))

    # Nearby labels folder without split.
    candidates.append(ip.parent.parent / "labels" / f"{stem}.txt")

    for c in candidates:
        try:
            cp = Path(c).expanduser().resolve()
        except Exception:
            continue
        if cp.exists() and cp.is_file():
            return cp
    return None


def _parse_pose_label_keypoints_px(
    label_path: Path,
    *,
    kp_names: List[str],
    W: int,
    H: int,
) -> Tuple[Dict[int, Tuple[int, int, float]], str]:
    if not label_path.exists() or not label_path.is_file():
        return {}, "label_missing"
    lines = [ln.strip() for ln in label_path.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
    if not lines:
        return {}, "label_empty"

    rows: List[List[float]] = []
    for ln in lines:
        try:
            vals = [float(x) for x in ln.split()]
        except Exception:
            continue
        if len(vals) >= 8:
            rows.append(vals)
    if not rows:
        return {}, "label_parse_failed"

    # Prefer aircraft class 0 row if available.
    chosen = rows[0]
    for r in rows:
        try:
            if int(round(float(r[0]))) == 0:
                chosen = r
                break
        except Exception:
            continue

    tail = chosen[5:]
    n_trip = len(tail) // 3
    n_kp = min(len(kp_names), n_trip)
    out: Dict[int, Tuple[int, int, float]] = {}
    for k in range(n_kp):
        xn = float(tail[3 * k + 0])
        yn = float(tail[3 * k + 1])
        vis = float(tail[3 * k + 2])
        if vis <= 0.0:
            continue
        if not (np.isfinite(xn) and np.isfinite(yn)):
            continue
        # YOLO normalized coords in [0,1]
        u = int(round(float(np.clip(xn, 0.0, 1.0)) * float(max(1, W - 1))))
        v = int(round(float(np.clip(yn, 0.0, 1.0)) * float(max(1, H - 1))))
        u = int(np.clip(u, 0, max(0, W - 1)))
        v = int(np.clip(v, 0, max(0, H - 1)))
        out[k] = (u, v, vis)
    return out, "ok"


def _build_h5_index_by_stem(source_root: str) -> Dict[str, List[Path]]:
    by_stem: Dict[str, List[Path]] = {}
    for hp in list_h5_paths(str(source_root)):
        p = Path(hp).expanduser().resolve()
        by_stem.setdefault(str(p.stem), []).append(p)
    return by_stem


def _load_xyz_scene_with_export_roll(
    *,
    unique_scene: str,
    h5_by_stem: Dict[str, List[Path]],
    yolo_pcd: Any,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int, int, str]:
    try:
        h5_stem, scene_name = yolo_pcd._parse_unique_scene_stem(unique_scene)
    except Exception as e:
        return None, None, 0, 0, f"scene_stem_parse_failed:{e}"
    matches = h5_by_stem.get(str(h5_stem), [])
    if not matches:
        return None, None, 0, 0, "h5_not_found"

    h5_path = matches[0]
    try:
        with open_h5_any(str(h5_path)) as f:
            H = int(f.attrs["height"])
            W = int(f.attrs["width"])
            if str(scene_name) not in f:
                return None, None, 0, 0, "scene_missing_in_h5"
            grp = f[str(scene_name)]
            if "points" not in grp:
                return None, None, 0, 0, "scene_points_missing"
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
            _rgb, xyz_hw3 = yolo_pcd._build_rgb_and_xyz(flat, cols, H, W)
            if xyz_hw3 is None:
                return None, None, 0, 0, "xyz_build_failed"
            mask_aircraft = yolo_pcd._extract_is_aircraft_mask(flat, cols, H, W)
            if mask_aircraft is not None:
                shift_cols = yolo_pcd._compute_export_like_roll(mask_aircraft)
                if int(shift_cols) != 0:
                    xyz_hw3 = np.roll(xyz_hw3, shift=int(shift_cols), axis=1)
                    mask_aircraft = np.roll(mask_aircraft, shift=int(shift_cols), axis=1)
            return np.asarray(xyz_hw3, dtype=np.float32), mask_aircraft, int(H), int(W), ""
    except Exception as e:
        return None, None, 0, 0, f"h5_read_failed:{e}"


def _export_gt_3d_from_2d(
    *,
    scenes_to_images: Dict[str, Path],
    yolo_pcd: Any,
    source_root: str,
    yaml_kp_names_path: str,
    kp_patch_radius: int,
    out_csv_path: Path,
) -> Tuple[int, int]:
    kp_names = _load_keypoint_names_from_yaml(yaml_kp_names_path)
    if not kp_names:
        return 0, 0

    h5_by_stem = _build_h5_index_by_stem(str(source_root))
    rows: List[List[Any]] = []
    total = 0
    ok_n = 0

    for unique_scene, img_path in sorted(scenes_to_images.items()):
        label_path = _infer_label_path_for_image(img_path)
        xyz_hw3, mask_aircraft, H, W, scene_reason = _load_xyz_scene_with_export_roll(
            unique_scene=unique_scene,
            h5_by_stem=h5_by_stem,
            yolo_pcd=yolo_pcd,
        )
        if xyz_hw3 is None:
            for k, kp_name in enumerate(kp_names):
                total += 1
                rows.append(
                    [
                        unique_scene,
                        str(img_path),
                        "" if label_path is None else str(label_path),
                        kp_name,
                        k,
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "scene_xyz_unavailable",
                        scene_reason,
                    ]
                )
            continue

        if label_path is None:
            for k, kp_name in enumerate(kp_names):
                total += 1
                rows.append(
                    [
                        unique_scene,
                        str(img_path),
                        "",
                        kp_name,
                        k,
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "label_missing",
                        "",
                    ]
                )
            continue

        kp_px_map, parse_reason = _parse_pose_label_keypoints_px(
            label_path=label_path,
            kp_names=kp_names,
            W=int(W),
            H=int(H),
        )
        for k, kp_name in enumerate(kp_names):
            total += 1
            if k not in kp_px_map:
                rows.append(
                    [
                        unique_scene,
                        str(img_path),
                        str(label_path),
                        kp_name,
                        k,
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "kp_missing_in_label",
                        parse_reason,
                    ]
                )
                continue
            u, v, vis = kp_px_map[k]
            p3 = yolo_pcd._sample_xyz_nearest(
                xyz_hw3=xyz_hw3,
                r0=int(v),
                c0=int(u),
                radius=int(kp_patch_radius),
                allow_global_fallback=False,
                mask_aircraft=mask_aircraft,
            )
            if p3 is None:
                rows.append(
                    [
                        unique_scene,
                        str(img_path),
                        str(label_path),
                        kp_name,
                        k,
                        int(u),
                        int(v),
                        f"{float(vis):.3f}",
                        "",
                        "",
                        "",
                        "no_depth_at_label",
                        "",
                    ]
                )
                continue
            ok_n += 1
            p3 = np.asarray(p3, dtype=np.float64).reshape(3)
            rows.append(
                [
                    unique_scene,
                    str(img_path),
                    str(label_path),
                    kp_name,
                    k,
                    int(u),
                    int(v),
                    f"{float(vis):.3f}",
                    f"{p3[0]:.6f}",
                    f"{p3[1]:.6f}",
                    f"{p3[2]:.6f}",
                    "ok",
                    "",
                ]
            )

    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with out_csv_path.open("w", newline="", encoding="utf-8") as f_csv:
        w = csv.writer(f_csv)
        w.writerow(
            [
                "unique_scene",
                "image_path",
                "label_path",
                "kp_name",
                "kp_index",
                "u_px",
                "v_px",
                "vis",
                "gt_x",
                "gt_y",
                "gt_z",
                "status",
                "reason",
            ]
        )
        for rr in rows:
            w.writerow(rr)
    return int(total), int(ok_n)


def _resolve_gt_point_for_pred_name(
    pred_name: str,
    gt_by_name: Dict[str, np.ndarray],
) -> Tuple[Optional[np.ndarray], str]:
    nm = str(pred_name or "").strip()
    if not nm:
        return None, "empty_name"

    # Exact name match first.
    if nm in gt_by_name:
        return np.asarray(gt_by_name[nm], dtype=np.float64).reshape(3), f"exact:{nm}"

    # Common aliases.
    aliases: Dict[str, List[str]] = {
        "engine_left_box_center": ["plane_engine_left", "engine_left", "left_engine"],
        "engine_right_box_center": ["plane_engine_right", "engine_right", "right_engine"],
    }
    for cand in aliases.get(nm, []):
        if cand in gt_by_name:
            return np.asarray(gt_by_name[cand], dtype=np.float64).reshape(3), f"alias:{cand}"

    # Derived midpoint for front wheels.
    if nm == "front_wheels_mid":
        left_names = ["plane_front_left_wheel_link", "front_left_wheel_link"]
        right_names = ["plane_front_right_wheel_link", "front_right_wheel_link"]
        pl = None
        pr = None
        for ln in left_names:
            if ln in gt_by_name:
                pl = np.asarray(gt_by_name[ln], dtype=np.float64).reshape(3)
                break
        for rn in right_names:
            if rn in gt_by_name:
                pr = np.asarray(gt_by_name[rn], dtype=np.float64).reshape(3)
                break
        if pl is not None and pr is not None:
            return 0.5 * (pl + pr), "derived:front_wheels_mid"

    return None, "gt_name_not_found"


def _compute_failed_kp_3d_errors(
    *,
    pcd_paths: List[Path],
    kp_conf_csv_path: Path,
    kp_passfail_csv_path: Path,
    warning_h5_root: str,
    fallback_kp_names: List[str],
    out_csv_path: Path,
    gt_3d_from_2d_csv_path: Optional[Path] = None,
) -> Tuple[int, int]:
    import view_pcd_dir as pcd_view

    if (not kp_passfail_csv_path.exists()) or (not kp_passfail_csv_path.is_file()):
        return 0, 0
    if (not kp_conf_csv_path.exists()) or (not kp_conf_csv_path.is_file()):
        return 0, 0

    scene_kp_names = pcd_view._infer_scene_keypoint_names_from_csv(kp_conf_csv_path)
    pcd_by_scene: Dict[str, Path] = {str(p.stem): p for p in pcd_paths}
    fail_rows: List[Dict[str, str]] = []
    with kp_passfail_csv_path.open("r", encoding="utf-8", newline="") as f_csv:
        for r in csv.DictReader(f_csv):
            kp_name = str(r.get("kp_name", "")).strip()
            if kp_name == "__scene__":
                continue
            if str(r.get("check_status", "")).strip().upper() != "FAIL":
                continue
            fail_rows.append(r)
    if not fail_rows:
        return 0, 0

    warning_state: Dict[str, Any] = {
        "scene_keypoints_cache": {},
        "warning_h5_root": str(warning_h5_root),
        "h5_path_cache": {},
    }
    gt2d_map: Dict[Tuple[str, str], np.ndarray] = {}
    if gt_3d_from_2d_csv_path is not None:
        gp = Path(gt_3d_from_2d_csv_path).expanduser().resolve()
        if gp.exists() and gp.is_file():
            with gp.open("r", encoding="utf-8", newline="") as f_csv:
                for rr in csv.DictReader(f_csv):
                    if str(rr.get("status", "")).strip().lower() != "ok":
                        continue
                    us = str(rr.get("unique_scene", "")).strip()
                    kn = str(rr.get("kp_name", "")).strip()
                    if not us or not kn:
                        continue
                    try:
                        gx = float(rr.get("gt_x", "nan"))
                        gy = float(rr.get("gt_y", "nan"))
                        gz = float(rr.get("gt_z", "nan"))
                    except Exception:
                        continue
                    if not (np.isfinite(gx) and np.isfinite(gy) and np.isfinite(gz)):
                        continue
                    gt2d_map[(us, kn)] = np.asarray([gx, gy, gz], dtype=np.float64)

    out_rows: List[List[Any]] = []
    total = 0
    matched = 0

    for r in fail_rows:
        unique_scene = str(r.get("unique_scene", "")).strip()
        kp_name = str(r.get("kp_name", "")).strip()
        if not unique_scene or not kp_name:
            continue
        total += 1

        pcd_path = pcd_by_scene.get(unique_scene, None)
        if pcd_path is None or (not pcd_path.exists()):
            out_rows.append(
                [unique_scene, "", kp_name, "", "", "", "", "", "", "", "", "pcd_not_found"]
            )
            continue

        names_scene = scene_kp_names.get(unique_scene, [])
        xyz = pcd_view._read_xyz(pcd_path, o3d=None)
        _, xyz_kp = pcd_view._split_main_and_keypoints(xyz, kpt_count=int(len(names_scene)))
        pred_by_name: Dict[str, np.ndarray] = {}
        if names_scene:
            n_map = min(int(xyz_kp.shape[0]), len(names_scene))
            for j in range(n_map):
                pred_by_name[str(names_scene[j])] = np.asarray(xyz_kp[j], dtype=np.float64).reshape(3)
            for j in range(n_map, int(xyz_kp.shape[0])):
                nm = (
                    str(fallback_kp_names[j])
                    if j < len(fallback_kp_names)
                    else f"K{j}"
                )
                pred_by_name[nm] = np.asarray(xyz_kp[j], dtype=np.float64).reshape(3)
        else:
            for j in range(int(xyz_kp.shape[0])):
                nm = (
                    str(fallback_kp_names[j])
                    if j < len(fallback_kp_names)
                    else f"K{j}"
                )
                pred_by_name[nm] = np.asarray(xyz_kp[j], dtype=np.float64).reshape(3)

        pred = pred_by_name.get(kp_name, None)
        if pred is None:
            out_rows.append(
                [unique_scene, str(pcd_path), kp_name, "", "", "", "", "", "", "", "", "pred_kp_not_found"]
            )
            continue

        gt = gt2d_map.get((unique_scene, kp_name), None)
        gt_src = "pseudo_2d"
        if gt is None:
            gt_names, gt_xyz, gt_reason = pcd_view._load_scene_keypoints_from_h5(
                unique_scene=unique_scene,
                warning_state=warning_state,
            )
            if gt_names is None or gt_xyz is None:
                out_rows.append(
                    [
                        unique_scene,
                        str(pcd_path),
                        kp_name,
                        f"{pred[0]:.6f}",
                        f"{pred[1]:.6f}",
                        f"{pred[2]:.6f}",
                        "",
                        "",
                        "",
                        "",
                        "",
                        f"gt_unavailable:{gt_reason}",
                    ]
                )
                continue
            gt_by_name = {
                str(gt_names[i]): np.asarray(gt_xyz[i], dtype=np.float64).reshape(3)
                for i in range(min(len(gt_names), int(gt_xyz.shape[0])))
            }
            gt, gt_src = _resolve_gt_point_for_pred_name(kp_name, gt_by_name)
        if gt is None:
            out_rows.append(
                [
                    unique_scene,
                    str(pcd_path),
                    kp_name,
                    f"{pred[0]:.6f}",
                    f"{pred[1]:.6f}",
                    f"{pred[2]:.6f}",
                    "",
                    "",
                    "",
                    "",
                    gt_src,
                    "gt_match_not_found",
                ]
            )
            continue

        d = float(np.linalg.norm(np.asarray(pred, dtype=np.float64) - np.asarray(gt, dtype=np.float64)))
        matched += 1
        out_rows.append(
            [
                unique_scene,
                str(pcd_path),
                kp_name,
                f"{pred[0]:.6f}",
                f"{pred[1]:.6f}",
                f"{pred[2]:.6f}",
                f"{gt[0]:.6f}",
                f"{gt[1]:.6f}",
                f"{gt[2]:.6f}",
                f"{d:.6f}",
                gt_src,
                "ok",
            ]
        )

    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with out_csv_path.open("w", newline="", encoding="utf-8") as f_csv:
        w = csv.writer(f_csv)
        w.writerow(
            [
                "unique_scene",
                "pcd_file",
                "kp_name",
                "pred_x",
                "pred_y",
                "pred_z",
                "gt_x",
                "gt_y",
                "gt_z",
                "error_l2_m",
                "gt_source",
                "status",
            ]
        )
        for rr in out_rows:
            w.writerow(rr)
    return total, matched


def _write_simple_csv(headers: List[str], rows: List[List[Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f_csv:
        w = csv.writer(f_csv)
        w.writerow(headers)
        for rr in rows:
            w.writerow(rr)


def _summarize_vals(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {
            "count": 0.0,
            "mean": float("nan"),
            "median": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "std": float("nan"),
        }
    arr = np.asarray(vals, dtype=np.float64).reshape(-1)
    return {
        "count": float(arr.shape[0]),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.quantile(arr, 0.90)),
        "p95": float(np.quantile(arr, 0.95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "std": float(np.std(arr)),
    }


def _export_failed_kp_3d_error_artifacts(
    *,
    fail_err_csv_path: Path,
    out_root: Path,
    top_n: int,
    plot_max_kp: int,
    plot_max_scene: int,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "total_rows": 0,
        "ok_rows": 0,
        "status_counts_csv": None,
        "gt_source_counts_csv": None,
        "by_keypoint_csv": None,
        "by_scene_csv": None,
        "top_csv": None,
        "plot_dir": None,
        "plots_created": 0,
        "plot_skip_reason": "",
    }
    src = Path(fail_err_csv_path).expanduser().resolve()
    if not src.exists() or not src.is_file():
        out["plot_skip_reason"] = "failed_keypoint_3d_error.csv missing"
        return out

    rows: List[Dict[str, str]] = []
    with src.open("r", encoding="utf-8", newline="") as f_csv:
        rows = list(csv.DictReader(f_csv))
    out["total_rows"] = int(len(rows))
    if not rows:
        out["plot_skip_reason"] = "failed_keypoint_3d_error.csv empty"
        return out

    status_counts: Dict[str, int] = {}
    gt_source_counts: Dict[str, int] = {}
    by_kp_vals: Dict[str, List[float]] = {}
    by_scene_vals: Dict[str, List[float]] = {}
    ok_rows: List[Dict[str, Any]] = []

    for r in rows:
        st = str(r.get("status", "")).strip() or "unknown"
        status_counts[st] = int(status_counts.get(st, 0)) + 1
        src_name = str(r.get("gt_source", "")).strip() or "unknown"
        gt_source_counts[src_name] = int(gt_source_counts.get(src_name, 0)) + 1
        if st.lower() != "ok":
            continue
        try:
            d = float(r.get("error_l2_m", "nan"))
        except Exception:
            continue
        if not np.isfinite(d):
            continue
        kp_name = str(r.get("kp_name", "")).strip() or "unknown_kp"
        scene = str(r.get("unique_scene", "")).strip() or "unknown_scene"
        by_kp_vals.setdefault(kp_name, []).append(float(d))
        by_scene_vals.setdefault(scene, []).append(float(d))
        rr = dict(r)
        rr["error_l2_m"] = float(d)
        ok_rows.append(rr)
    out["ok_rows"] = int(len(ok_rows))

    status_rows = [[k, int(v)] for k, v in sorted(status_counts.items(), key=lambda kv: kv[0])]
    gt_src_rows = [[k, int(v)] for k, v in sorted(gt_source_counts.items(), key=lambda kv: kv[0])]

    by_kp_rows: List[List[Any]] = []
    for kp_name, vals in sorted(by_kp_vals.items(), key=lambda kv: (kv[0])):
        s = _summarize_vals(vals)
        by_kp_rows.append(
            [
                kp_name,
                int(s["count"]),
                f"{s['mean']:.6f}",
                f"{s['median']:.6f}",
                f"{s['p90']:.6f}",
                f"{s['p95']:.6f}",
                f"{s['min']:.6f}",
                f"{s['max']:.6f}",
                f"{s['std']:.6f}",
            ]
        )
    by_kp_rows.sort(key=lambda rr: float(rr[2]), reverse=True)

    by_scene_rows: List[List[Any]] = []
    for scene_name, vals in sorted(by_scene_vals.items(), key=lambda kv: (kv[0])):
        s = _summarize_vals(vals)
        by_scene_rows.append(
            [
                scene_name,
                int(s["count"]),
                f"{s['mean']:.6f}",
                f"{s['median']:.6f}",
                f"{s['p90']:.6f}",
                f"{s['p95']:.6f}",
                f"{s['min']:.6f}",
                f"{s['max']:.6f}",
                f"{s['std']:.6f}",
            ]
        )
    by_scene_rows.sort(key=lambda rr: float(rr[2]), reverse=True)

    top_keep = max(1, int(top_n))
    top_rows_src = sorted(
        ok_rows,
        key=lambda rr: float(rr.get("error_l2_m", 0.0)),
        reverse=True,
    )[:top_keep]
    top_rows: List[List[Any]] = []
    for rr in top_rows_src:
        top_rows.append(
            [
                str(rr.get("unique_scene", "")),
                str(rr.get("kp_name", "")),
                f"{float(rr.get('error_l2_m', 0.0)):.6f}",
                str(rr.get("gt_source", "")),
                str(rr.get("pcd_file", "")),
            ]
        )

    status_csv = Path(out_root) / "failed_keypoint_3d_error_status_counts.csv"
    gt_src_csv = Path(out_root) / "failed_keypoint_3d_error_gt_source_counts.csv"
    by_kp_csv = Path(out_root) / "failed_keypoint_3d_error_by_keypoint.csv"
    by_scene_csv = Path(out_root) / "failed_keypoint_3d_error_by_scene.csv"
    top_csv = Path(out_root) / "failed_keypoint_3d_error_top.csv"

    _write_simple_csv(["status", "count"], status_rows, status_csv)
    _write_simple_csv(["gt_source", "count"], gt_src_rows, gt_src_csv)
    _write_simple_csv(
        ["kp_name", "count", "mean_error_m", "median_error_m", "p90_error_m", "p95_error_m", "min_error_m", "max_error_m", "std_error_m"],
        by_kp_rows,
        by_kp_csv,
    )
    _write_simple_csv(
        ["unique_scene", "count", "mean_error_m", "median_error_m", "p90_error_m", "p95_error_m", "min_error_m", "max_error_m", "std_error_m"],
        by_scene_rows,
        by_scene_csv,
    )
    _write_simple_csv(
        ["unique_scene", "kp_name", "error_l2_m", "gt_source", "pcd_file"],
        top_rows,
        top_csv,
    )

    out["status_counts_csv"] = status_csv
    out["gt_source_counts_csv"] = gt_src_csv
    out["by_keypoint_csv"] = by_kp_csv
    out["by_scene_csv"] = by_scene_csv
    out["top_csv"] = top_csv

    if not ok_rows:
        out["plot_skip_reason"] = "no status=ok rows with numeric error"
        return out

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        out["plot_skip_reason"] = f"matplotlib_unavailable:{type(e).__name__}"
        return out

    vis_dir = Path(out_root) / "failure_distance_vis"
    vis_dir.mkdir(parents=True, exist_ok=True)
    out["plot_dir"] = vis_dir
    plots_created = 0

    all_vals = [float(rr["error_l2_m"]) for rr in ok_rows]
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(all_vals, bins=40, color="#2f6db2", edgecolor="white")
        ax.set_title("Failed keypoint 3D error distribution")
        ax.set_xlabel("L2 error (m)")
        ax.set_ylabel("Count")
        ax.grid(True, which="major", axis="both", linestyle="--", alpha=0.35)
        fig.tight_layout()
        fig.savefig(vis_dir / "error_hist_all.png", dpi=160)
        plt.close(fig)
        plots_created += 1
    except Exception:
        pass

    kp_items = [(k, v) for k, v in by_kp_vals.items() if v]
    kp_items.sort(key=lambda kv: float(np.mean(np.asarray(kv[1], dtype=np.float64))), reverse=True)
    kp_items = kp_items[: max(1, int(plot_max_kp))]
    if kp_items:
        kp_names = [k for k, _ in kp_items]
        kp_means = [float(np.mean(np.asarray(v, dtype=np.float64))) for _, v in kp_items]
        try:
            fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(kp_names) + 1.5)))
            y = np.arange(len(kp_names))
            ax.barh(y, kp_means, color="#d95f02")
            ax.set_yticks(y)
            ax.set_yticklabels(kp_names, fontsize=9)
            ax.invert_yaxis()
            ax.set_xlabel("Mean error (m)")
            ax.set_title("Mean failed 3D error by keypoint")
            ax.grid(True, which="major", axis="x", linestyle="--", alpha=0.35)
            fig.tight_layout()
            fig.savefig(vis_dir / "error_mean_by_keypoint.png", dpi=160)
            plt.close(fig)
            plots_created += 1
        except Exception:
            pass

        try:
            fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(kp_names) + 1.5)))
            data = [np.asarray(v, dtype=np.float64) for _, v in kp_items]
            ax.boxplot(data, vert=False, tick_labels=kp_names, showfliers=True)
            ax.set_xlabel("Error (m)")
            ax.set_title("Failed 3D error boxplot by keypoint")
            ax.grid(True, which="major", axis="x", linestyle="--", alpha=0.35)
            fig.tight_layout()
            fig.savefig(vis_dir / "error_boxplot_by_keypoint.png", dpi=160)
            plt.close(fig)
            plots_created += 1
        except Exception:
            pass

    scene_items = [(k, v) for k, v in by_scene_vals.items() if v]
    scene_items.sort(key=lambda kv: float(np.mean(np.asarray(kv[1], dtype=np.float64))), reverse=True)
    scene_items = scene_items[: max(1, int(plot_max_scene))]
    if scene_items:
        scene_names = [k for k, _ in scene_items]
        scene_means = [float(np.mean(np.asarray(v, dtype=np.float64))) for _, v in scene_items]
        try:
            fig, ax = plt.subplots(figsize=(12, 4.5))
            x = np.arange(len(scene_names))
            ax.bar(x, scene_means, color="#4daf4a")
            ax.set_xticks(x)
            ax.set_xticklabels(scene_names, rotation=75, ha="right", fontsize=8)
            ax.set_ylabel("Mean error (m)")
            ax.set_title("Top scenes by mean failed 3D error")
            ax.grid(True, which="major", axis="y", linestyle="--", alpha=0.35)
            fig.tight_layout()
            fig.savefig(vis_dir / "error_mean_by_scene_top.png", dpi=160)
            plt.close(fig)
            plots_created += 1
        except Exception:
            pass

    out["plots_created"] = int(plots_created)
    if plots_created <= 0 and not out["plot_skip_reason"]:
        out["plot_skip_reason"] = "plot_generation_failed"
    return out


def _to_float_or_nan(v: Any) -> float:
    try:
        x = float(v)
    except Exception:
        return float("nan")
    return x if np.isfinite(x) else float("nan")


def _export_warning_box_distance_csv(
    *,
    kp_passfail_csv_path: Path,
    out_csv_path: Path,
) -> Tuple[int, int, int, float]:
    src = Path(kp_passfail_csv_path).expanduser().resolve()
    if not src.exists() or not src.is_file():
        return 0, 0, 0, 0.0

    rows_out: List[List[Any]] = []
    total = 0
    with_metrics = 0
    outside_n = 0
    outside_sum = 0.0

    with src.open("r", encoding="utf-8", newline="") as f_csv:
        for r in csv.DictReader(f_csv):
            kp_name = str(r.get("kp_name", "")).strip()
            if not kp_name or kp_name == "__scene__":
                continue
            total += 1
            status = str(r.get("check_status", "")).strip().upper() or "UNKNOWN"
            unique_scene = str(r.get("unique_scene", "")).strip()
            box_key = str(r.get("box_key", "")).strip()

            ax = _to_float_or_nan(r.get("abs_local_x", ""))
            ay = _to_float_or_nan(r.get("abs_local_y", ""))
            az = _to_float_or_nan(r.get("abs_local_z", ""))
            hx = _to_float_or_nan(r.get("half_x", ""))
            hy = _to_float_or_nan(r.get("half_y", ""))
            hz = _to_float_or_nan(r.get("half_z", ""))

            if not all(np.isfinite(v) for v in [ax, ay, az, hx, hy, hz]):
                rows_out.append(
                    [
                        unique_scene,
                        kp_name,
                        box_key,
                        status,
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "missing_local_or_half_dims",
                    ]
                )
                continue

            dx = float(ax - hx)
            dy = float(ay - hy)
            dz = float(az - hz)
            ox = float(max(dx, 0.0))
            oy = float(max(dy, 0.0))
            oz = float(max(dz, 0.0))
            outside_dist = float(np.sqrt(ox * ox + oy * oy + oz * oz))
            inside_margin = float(min(hx - ax, hy - ay, hz - az))
            inside_ok = bool((dx <= 0.0) and (dy <= 0.0) and (dz <= 0.0))
            signed_dist = float(-inside_margin) if inside_ok else float(outside_dist)

            with_metrics += 1
            if outside_dist > 0.0:
                outside_n += 1
                outside_sum += outside_dist

            rows_out.append(
                [
                    unique_scene,
                    kp_name,
                    box_key,
                    status,
                    f"{outside_dist:.6f}",
                    f"{signed_dist:.6f}",
                    f"{inside_margin:.6f}",
                    f"{dx:.6f}",
                    f"{dy:.6f}",
                    f"{dz:.6f}",
                    int(inside_ok),
                    "ok",
                ]
            )

    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with out_csv_path.open("w", newline="", encoding="utf-8") as f_csv:
        w = csv.writer(f_csv)
        w.writerow(
            [
                "unique_scene",
                "kp_name",
                "box_key",
                "check_status",
                "outside_distance_m",
                "signed_distance_m",
                "inside_margin_m",
                "delta_x_m",
                "delta_y_m",
                "delta_z_m",
                "inside_box",
                "status",
            ]
        )
        for rr in rows_out:
            w.writerow(rr)

    outside_mean = float(outside_sum / outside_n) if outside_n > 0 else 0.0
    return int(total), int(with_metrics), int(outside_n), float(outside_mean)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run YOLO on one test image, export PCD, then optional visualization and warning-box checks."
        )
    )
    p.add_argument(
        "--image-path",
        type=str,
        default=DEFAULT_IMAGE_PATH,
        help="Test image path OR directory path of test images",
    )
    p.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS_PATH, help="YOLO .pt weights or weights dir")
    p.add_argument("--source", type=str, default=DEFAULT_SOURCE_H5_ROOT, help="H5 root used to resolve matching scene")
    p.add_argument("--out", type=str, default=DEFAULT_OUT_DIR, help="Output directory for generated PCD/debug files")
    p.add_argument("--yaml-kp-names", type=str, default=DEFAULT_YAML_KP_NAMES, help="aircraft_pose.yaml for keypoint labels")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import test_yolo_pose_from_h5_weights_to_pcd as yolo_pcd
        import view_pcd_dir as pcd_view
    except Exception as e:
        raise RuntimeError(
            "Failed to import pipeline modules. Run with your project venv "
            "(e.g. /home/femi/Benchmarking_framework/.venv/bin/python)."
        ) from e

    image_path = str(args.image_path or "").strip()
    weights = str(args.weights or "").strip()
    source = str(args.source or "").strip()
    out_dir = str(args.out or "").strip()
    yaml_kp_names = str(args.yaml_kp_names or "").strip()

    if not image_path:
        raise RuntimeError("Please set --image-path (or DEFAULT_IMAGE_PATH in code).")
    if not weights:
        raise RuntimeError("Please set --weights (or DEFAULT_WEIGHTS_PATH in code).")
    if not source:
        raise RuntimeError("Please set --source (or DEFAULT_SOURCE_H5_ROOT in code).")
    if not out_dir:
        raise RuntimeError("Please set --out (or DEFAULT_OUT_DIR in code).")

    image_path_obj = Path(image_path).expanduser()
    is_dir_mode = bool(image_path_obj.exists() and image_path_obj.is_dir())
    resolved_image: Path | None = None
    image_dir_path: Path | None = None
    if is_dir_mode:
        image_dir_path = image_path_obj.resolve()
    else:
        resolved_image = yolo_pcd._resolve_image_path_with_split_fallback(image_path)
    scenes_to_images: Dict[str, Path] = {}
    if is_dir_mode:
        for ip in yolo_pcd._collect_images_from_dir(image_dir_path):
            scenes_to_images[str(Path(ip).stem)] = Path(ip).resolve()
    else:
        if resolved_image is not None:
            scenes_to_images[str(resolved_image.stem)] = resolved_image.resolve()

    out_root = Path(out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    kp_conf_csv_path = out_root / "keypoint_confidence.csv"
    bbox_cov_csv_path = out_root / "bbox_aircraft_coverage.csv"
    kp_passfail_csv_path = out_root / "keypoint_pass_fail_confidence.csv"
    warn_box_dist_csv_path = out_root / "warning_box_distance.csv"
    fail_kp_err_csv_path = out_root / "failed_keypoint_3d_error.csv"
    gt_3d_from_2d_csv_path = out_root / "gt_3d_from_2d.csv"

    print("[pipeline] Step 1/2: YOLO -> PCD export")
    if is_dir_mode:
        print(f"[pipeline] image-dir: {image_dir_path}")
    else:
        print(f"[pipeline] image: {resolved_image}")
    print(f"[pipeline] weights: {weights}")
    print(f"[pipeline] source: {source}")
    print(f"[pipeline] out: {out_root}")
    print(
        "[pipeline] bbox full-aircraft check: "
        f"enabled={bool(CHECK_BBOX_COVERAGE)} thr={float(BBOX_FULL_THR):.3f}"
    )

    yolo_pcd.run(
        source=source,
        weights=weights,
        out_dir=str(out_root),
        max_h5_files=None,
        imgsz=int(IMG_SIZE),
        conf=float(YOLO_CONF),
        device=str(DEVICE),
        save_img=bool(SAVE_DEBUG_IMAGE),
        yaml_kp_names=yaml_kp_names,
        kp_conf=float(KP_CONF_THR),
        kp_patch_radius=int(KP_PATCH_RADIUS),
        show_3d=False,
        max_vis_scenes=1,
        image_path=(None if is_dir_mode else str(resolved_image)),
        image_dir=(str(image_dir_path) if is_dir_mode else None),
        print_kp_conf=True,
        kp_conf_csv=str(kp_conf_csv_path),
        check_bbox_coverage=bool(CHECK_BBOX_COVERAGE),
        bbox_full_thr=float(BBOX_FULL_THR),
        bbox_cov_csv=(str(bbox_cov_csv_path) if bool(CHECK_BBOX_COVERAGE) else ""),
    )
    if bool(EXPORT_GT_3D_FROM_2D):
        try:
            gt_total, gt_ok = _export_gt_3d_from_2d(
                scenes_to_images=scenes_to_images,
                yolo_pcd=yolo_pcd,
                source_root=str(source),
                yaml_kp_names_path=str(yaml_kp_names),
                kp_patch_radius=int(KP_PATCH_RADIUS),
                out_csv_path=gt_3d_from_2d_csv_path,
            )
            print(
                "[pipeline] gt_3d_from_2d csv: "
                f"{gt_3d_from_2d_csv_path} "
                f"(rows={int(gt_total)} ok={int(gt_ok)})"
            )
        except Exception as e:
            print(f"[warn] gt_3d_from_2d export failed: {type(e).__name__}: {e}")
    if bool(CHECK_BBOX_COVERAGE):
        print(f"[pipeline] bbox coverage csv: {bbox_cov_csv_path}")
        if bbox_cov_csv_path.exists() and bbox_cov_csv_path.is_file():
            with bbox_cov_csv_path.open("r", encoding="utf-8", newline="") as f_csv:
                rows = list(csv.DictReader(f_csv))
            if not rows:
                print("[bbox-passfail] status=UNKNOWN reason=empty_csv")
            else:
                for row in rows:
                    unique_scene = str(row.get("unique_scene", "")).strip() or "unknown_scene"
                    status = str(row.get("bbox_status", "UNKNOWN")).strip().upper() or "UNKNOWN"
                    reason = str(row.get("bbox_reason", "")).strip() or "n/a"
                    x1 = str(row.get("bbox_x1", "")).strip()
                    y1 = str(row.get("bbox_y1", "")).strip()
                    x2 = str(row.get("bbox_x2", "")).strip()
                    y2 = str(row.get("bbox_y2", "")).strip()
                    rec = str(row.get("aircraft_recall", "")).strip()
                    bbox_txt = f"({x1},{y1},{x2},{y2})" if all([x1, y1, x2, y2]) else "(n/a)"
                    rec_txt = rec if rec else "n/a"
                    print(
                        f"[bbox-passfail] {unique_scene}: status={status} "
                        f"reason={reason} bbox={bbox_txt} recall={rec_txt}"
                    )
        else:
            print("[bbox-passfail] status=UNKNOWN reason=missing_bbox_csv")

    pcd_paths = sorted(out_root.glob("*.pcd"))
    if not pcd_paths:
        raise RuntimeError(f"No output PCD found under: {out_root}")

    print("[pipeline] Step 2/2: PCD view/check")
    print(f"[pipeline] code-toggle VISUALIZE_PCD={bool(VISUALIZE_PCD)}")
    print(f"[pipeline] code-toggle RUN_WARNING_CHECK={bool(RUN_WARNING_CHECK)}")
    print(f"[pipeline] code-toggle RUN_WARNING_PASS_FAIL={bool(RUN_WARNING_PASS_FAIL)}")

    fallback_kp_names = pcd_view._parse_name_csv(WARNING_FALLBACK_KP_NAMES)

    pcd_view._view_files(
        paths=pcd_paths,
        show_axes=bool(SHOW_AXES),
        visualize=bool(VISUALIZE_PCD),
        kpt_count=int(KPT_COUNT),
        kpt_radius=float(KPT_RADIUS_M),
        warning_check_enabled=bool(RUN_WARNING_CHECK),
        warning_pass_fail_enabled=bool(RUN_WARNING_PASS_FAIL and RUN_WARNING_CHECK),
        warning_keypoint_csv=kp_conf_csv_path,
        warning_profile_csv=str(pcd_view.WARNING_PROFILE_CSV),
        warning_yaml_column=str(pcd_view.WARNING_YAML_COLUMN),
        warning_yaml_root=str(pcd_view.WARNING_YAML_ROOT),
        warning_yaml_relpath=str(pcd_view.WARNING_YAML_RELPATH),
        warning_target_level=int(pcd_view.WARNING_TARGET_LEVEL),
        warning_box_scale=float(WARNING_BOX_SCALE),
        warning_fallback_kp_names=fallback_kp_names,
        warning_h5_root=str(source),
        use_scene_h5_transform=bool(USE_SCENE_H5_TRANSFORM),
        warning_kp_passfail_csv=kp_passfail_csv_path,
        warning_conf_threshold=float(KP_CONF_THR),
    )
    if bool(RUN_WARNING_CHECK):
        print(f"[pipeline] keypoint pass/fail confidence csv: {kp_passfail_csv_path}")
        if bool(EXPORT_WARNING_BOX_DISTANCE):
            try:
                dist_total, dist_ok, dist_out_n, dist_out_mean = _export_warning_box_distance_csv(
                    kp_passfail_csv_path=kp_passfail_csv_path,
                    out_csv_path=warn_box_dist_csv_path,
                )
                print(
                    "[pipeline] warning-box distance csv: "
                    f"{warn_box_dist_csv_path} "
                    f"(rows={int(dist_total)} metric_rows={int(dist_ok)})"
                )
                print(
                    "[pipeline] warning-box outside distance: "
                    f"outside_rows={int(dist_out_n)} "
                    f"outside_mean_m={float(dist_out_mean):.4f}"
                )
            except Exception as e:
                print(f"[warn] warning-box distance export failed: {type(e).__name__}: {e}")
        if bool(COMPARE_FAIL_KP_3D_ERROR):
            try:
                total_fail, matched_fail = _compute_failed_kp_3d_errors(
                    pcd_paths=pcd_paths,
                    kp_conf_csv_path=kp_conf_csv_path,
                    kp_passfail_csv_path=kp_passfail_csv_path,
                    warning_h5_root=str(source),
                    fallback_kp_names=fallback_kp_names,
                    out_csv_path=fail_kp_err_csv_path,
                    gt_3d_from_2d_csv_path=(
                        gt_3d_from_2d_csv_path
                        if bool(USE_GT_3D_FROM_2D_FOR_FAIL_DISTANCE)
                        else None
                    ),
                )
                print(
                    "[pipeline] failed keypoint 3D error csv: "
                    f"{fail_kp_err_csv_path} "
                    f"(failed_rows={int(total_fail)} matched_gt={int(matched_fail)})"
                )
                if bool(EXPORT_FAIL_DISTANCE_VIS):
                    vis_info = _export_failed_kp_3d_error_artifacts(
                        fail_err_csv_path=fail_kp_err_csv_path,
                        out_root=out_root,
                        top_n=int(FAIL_DISTANCE_TOP_N),
                        plot_max_kp=int(FAIL_DISTANCE_PLOT_MAX_KP),
                        plot_max_scene=int(FAIL_DISTANCE_PLOT_MAX_SCENE),
                    )
                    print(
                        "[pipeline] failed 3D error summaries: "
                        f"by_kp={vis_info.get('by_keypoint_csv')} "
                        f"by_scene={vis_info.get('by_scene_csv')} "
                        f"top={vis_info.get('top_csv')}"
                    )
                    print(
                        "[pipeline] failed 3D error counts: "
                        f"total_rows={int(vis_info.get('total_rows', 0))} "
                        f"ok_rows={int(vis_info.get('ok_rows', 0))}"
                    )
                    if vis_info.get("plot_dir"):
                        print(
                            "[pipeline] failed 3D error plots: "
                            f"dir={vis_info.get('plot_dir')} "
                            f"created={int(vis_info.get('plots_created', 0))}"
                        )
                    elif vis_info.get("plot_skip_reason"):
                        print(
                            "[pipeline] failed 3D error plots skipped: "
                            f"{vis_info.get('plot_skip_reason')}"
                        )
            except Exception as e:
                print(f"[warn] failed keypoint 3D error compute failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
