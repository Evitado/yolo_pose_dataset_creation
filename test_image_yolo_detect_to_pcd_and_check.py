#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Detection pipeline:
1) Run YOLO-detect on one test image (or all images in a directory)
2) Backproject aircraft bbox to PCD using matching H5 scene
3) Convert engine bboxes to 3D proxy keypoints (bbox centers) for warning-box checks
4) Optionally visualize + run warning-box pass/fail checks

Expected image filename stem format:
  <h5_stem>__<scene_name>
Example:
  movement_737_900er__2025-09-11T19-56-15__scene_000.png
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import h5py
import numpy as np

from io_helpers import list_h5_paths, open_h5_any


# =========================
# Code-level toggles (edit here)
# =========================movement_777_300__2025-08-28T05-01-56__scene_004
VISUALIZE_PCD: bool = True
VISUALIZE_FAILED_SCENES_ONLY: bool = False
RUN_WARNING_CHECK: bool = True
RUN_WARNING_PASS_FAIL: bool = True
USE_SCENE_H5_TRANSFORM: bool = True
SHOW_AXES: bool = False
CHECK_BBOX_COVERAGE: bool = True
SAVE_DEBUG_IMAGE: bool = True
INCLUDE_FRONT_PROXY_KEYPOINT: bool = False
USE_FRONT_BBOX_FOR_FRONT_PROXY: bool = True
FRONT_PROXY_KEYPOINT_NAME: str = "front_wheels_mid"
DRAW_PROXY_KEYPOINTS_ON_DEBUG_IMAGE: bool = False
SAVE_ENGINE_REGION_PCD: bool = True
SHOW_ENGINE_REGION_OVERLAY_3D: bool = True
USE_ENGINE_REGION_RATIO_FOR_PASSFAIL: bool = True
ENGINE_REGION_INSIDE_RATIO_THR: float = 0.80
FAILED_SCENES_MODE: str = "list"  # off | list | open
OPEN_FAILED_LIMIT: int = 12
RUN_REGION_HYPOTHESIS: bool = True
REGION_HYPOTHESIS_MIN_ENGINES: int = 1
REGION_HYPOTHESIS_REQUIRE_FRONT_GEAR: bool = True
REGION_HYPOTHESIS_PYTHON: str = ""
VISUALIZE_REGION_HYPOTHESIS_OUTPUTS: bool = True
VISUALIZE_REGION_HYPOTHESIS_SPLIT_WINDOWS: bool = True
REGION_HYPOTHESIS_VIS_MAX_SCENES: int = 0  # 0 = all
REGION_HYPOTHESIS_CENTER_SPHERE_RADIUS: float = 0.35
RUN_MERGER_POSE: bool = True
MERGER_POSE_DETECT_ALL_PARTS: bool = False
MERGER_POSE_REQUIRE_FINAL_GRAPH: bool = True
MERGER_POSE_USE_REGION_PCD_ONLY: bool = True
MERGER_POSE_PYTHON: str = ""
MERGER_POSE_VISUALIZE: bool = True
MERGER_POSE_SHOW_DETECTED_PARTS: bool = True
MERGER_POSE_LOG_MODE: str = "minimal"  # off | minimal | full
SAVE_MERGER_POSE_FIXED_CSV: bool = True
MERGER_POSE_FIXED_CSV_PATH: str = (
    "/home/femi/prof/outputs/csv/nose_engine_redesigned14_yolo_info.csv"
)
STRICT_PER_SCENE_CHAINING: bool = True
# Fine-grained visualization toggles (edit on/off here)
# Only final merger-pose visualization enabled by default.
VIS_CHAIN_PCD_VIEW: bool = True
VIS_CHAIN_ENGINE_REGION_DETECTOR: bool = False
VIS_CHAIN_REGION_HYPOTHESIS: bool = True
VIS_CHAIN_MERGER_POSE: bool = True
VIS_BATCH_PCD_VIEW: bool = False
VIS_BATCH_ENGINE_REGION_DETECTOR: bool = False
VIS_BATCH_REGION_HYPOTHESIS: bool = False
VIS_BATCH_MERGER_POSE: bool = True
SUPPRESS_PRINTS: bool = True
RUN_RANSAC_GROUND_REMOVAL: bool = True
RANSAC_GROUND_DIST_THR: float = 0.12
RANSAC_GROUND_MAX_ITERS: int = 400
RANSAC_GROUND_MIN_INLIER_RATIO: float = 0.08
RANSAC_GROUND_MIN_ABS_Z: float = 0.65
RANSAC_KEEP_ORIGINAL_IF_EMPTY: bool = True
DEFAULT_AIRCRAFT_PIPELINE_ROOT: str = "/home/femi/prof/geometric_aware_aircraft_pose_estimation_pipeline"

# Model / conversion settingsz
IMG_SIZE: int = 1024
YOLO_CONF: float = 0.05
DEVICE: str = "0"
BBOX_FULL_THR: float = 0.995
ENGINE_CENTER_PATCH_RADIUS: int = 3
BBOX_MARGIN_M: float = 1.5
PART_REGION_BBOX_MARGIN_M: float = 0.0
PART_REGION_Z_EXPAND_M: float = 3.0
FRONT_REGION_BBOX_MARGIN_M: float = 3.0

# Default paths
DEFAULT_IMAGE_PATH: str = (
    "/home/femi/yolo_pose_dataset_creation/aircraft_engine_det_test_edited/images/test"
)
DEFAULT_WEIGHTS_PATH: str = (
    "/home/femi/yolo_pose_dataset_creation/runs/detect/aircraft_engine_frontgear_det-5/weights/best.pt"
)
DEFAULT_SOURCE_H5_ROOT: str = "/home/femi/Benchmarking_framework/Data/warning_b_test_h5"
DEFAULT_OUT_DIR: str = str(Path.cwd() / "pcd_from_yolo_detect2")

# Class ids in detection model
CLASS_AIRCRAFT: int = 0
CLASS_ENGINE_LEFT: int = 1
CLASS_ENGINE_RIGHT: int = 2
CLASS_FRONT_GEAR: int = 3
FORCE_ENGINE_SIDE_BY_IMAGE_X: bool = False
TRY_H5_RGB_DETECTION_FALLBACK: bool = True
H5_RGB_FALLBACK_MIN_AIRCRAFT_RECALL: float = 0.92
H5_RGB_FALLBACK_IF_NO_ENGINE_BBOX: bool = True


@dataclass
class SceneRequest:
    scene_name: str
    image_path: Path
    image_bgr: np.ndarray


def _switch_default(v: bool) -> str:
    return "on" if bool(v) else "off"


def _to_bool_switch(raw: str, fallback: bool) -> bool:
    s = str(raw or "").strip().lower()
    if s in {"on", "true", "1", "yes", "y"}:
        return True
    if s in {"off", "false", "0", "no", "n"}:
        return False
    return bool(fallback)
def _write_labeled_region_centers_csv(
    *,
    vis_rows: List[Dict[str, Any]],
    out_csv: Path,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["unique_scene", "part", "candidate_index", "x", "y", "z"])

        for vr in vis_rows:
            scene = str(vr.get("scene", "")).strip()
            if not scene:
                continue

            for i, c in enumerate(vr.get("left_centers", [])):
                c = np.asarray(c, dtype=np.float64).reshape(3)
                w.writerow([scene, "engine_left", i, c[0], c[1], c[2]])

            for i, c in enumerate(vr.get("right_centers", [])):
                c = np.asarray(c, dtype=np.float64).reshape(3)
                w.writerow([scene, "engine_right", i, c[0], c[1], c[2]])

            for i, c in enumerate(vr.get("front_nose_centers", [])):
                c = np.asarray(c, dtype=np.float64).reshape(3)
                w.writerow([scene, "nose_gear", i, c[0], c[1], c[2]])

            for i, c in enumerate(vr.get("front_main_centers", [])):
                c = np.asarray(c, dtype=np.float64).reshape(3)
                w.writerow([scene, "main_gear", i, c[0], c[1], c[2]])

def _normalize_log_mode(raw: str, fallback: str = "full") -> str:
    s = str(raw or "").strip().lower()
    if s in {"off", "minimal", "full"}:
        return s
    fb = str(fallback or "").strip().lower()
    if fb in {"off", "minimal", "full"}:
        return fb
    return "full"


def _emit_console(msg: str) -> None:
    try:
        sys.stdout.write(f"{str(msg)}\n")
        sys.stdout.flush()
    except Exception:
        pass


def _configure_print_suppression(enabled: bool) -> None:
    if not bool(enabled):
        return
    import builtins

    builtins.print = lambda *args, **kwargs: None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run YOLO-detect on test image(s), export aircraft-bbox PCD, "
            "and run optional warning-box checks."
        )
    )
    p.add_argument(
        "--image-path",
        type=str,
        default=DEFAULT_IMAGE_PATH,
        help="Single image path or directory path",
    )
    p.add_argument("--image-dir", type=str, default="", help="Directory containing test images")
    p.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS_PATH, help="YOLO detect .pt weights")
    p.add_argument("--source", type=str, default=DEFAULT_SOURCE_H5_ROOT, help="Root folder containing source H5 files")
    p.add_argument("--out", type=str, default=DEFAULT_OUT_DIR, help="Output directory")
    p.add_argument(
        "--quiet",
        type=str,
        default=_switch_default(SUPPRESS_PRINTS),
        choices=["on", "off"],
        help="Suppress most console print output from this script.",
    )

    p.add_argument("--imgsz", type=int, default=int(IMG_SIZE), help="YOLO inference image size")
    p.add_argument("--conf", type=float, default=float(YOLO_CONF), help="YOLO confidence threshold")
    p.add_argument("--device", type=str, default=str(DEVICE), help="YOLO device (e.g. 0 or cpu)")
    p.add_argument(
        "--bbox-full-thr",
        type=float,
        default=float(BBOX_FULL_THR),
        help="Aircraft-mask coverage threshold for bbox PASS/FAIL",
    )
    p.add_argument(
        "--engine-center-patch-radius",
        type=int,
        default=int(ENGINE_CENTER_PATCH_RADIUS),
        help="Patch radius used to sample nearest valid xyz at engine bbox center",
    )
    p.add_argument(
        "--bbox-margin-m",
        type=float,
        default=float(BBOX_MARGIN_M),
        help=(
            "Include additional points within this Euclidean XYZ radius (meters) "
            "from bbox-derived seed points. Use 0 for strict bbox-only extraction."
        ),
    )
    p.add_argument(
        "--part-bbox-margin-m",
        type=float,
        default=float(PART_REGION_BBOX_MARGIN_M),
        help=(
            "Metric expansion radius used only for part regions "
            "(engine_left/right/front_gear). Set 0 for strict part-region extraction."
        ),
    )
    p.add_argument(
        "--part-region-z-expand-m",
        type=float,
        default=float(PART_REGION_Z_EXPAND_M),
        help=(
            "Vertical expansion for each part region pointcloud: "
            "include scene points within [zmin-Δ, zmax+Δ] while keeping region XY bounds."
        ),
    )
    p.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Maximum images to process (0 = all)",
    )
    p.add_argument(
        "--include-front-proxy",
        type=str,
        default=_switch_default(INCLUDE_FRONT_PROXY_KEYPOINT),
        choices=["on", "off"],
        help="Append front proxy keypoint only when detected front bbox is available",
    )
    p.add_argument(
        "--use-front-bbox-proxy",
        type=str,
        default=_switch_default(USE_FRONT_BBOX_FOR_FRONT_PROXY),
        choices=["on", "off"],
        help="When front proxy is enabled, require detected front bbox center (no fallback)",
    )
    p.add_argument(
        "--front-class-id",
        type=int,
        default=int(CLASS_FRONT_GEAR),
        help="Detection class id for front gear bbox",
    )
    p.add_argument(
        "--force-engine-side-by-image-x",
        type=str,
        default=_switch_default(FORCE_ENGINE_SIDE_BY_IMAGE_X),
        choices=["on", "off"],
        help=(
            "Normalize engine_left/right by image x-position "
            "(and aircraft center when only one engine bbox is present)."
        ),
    )
    p.add_argument(
        "--front-proxy-name",
        type=str,
        default=str(FRONT_PROXY_KEYPOINT_NAME),
        help="Keypoint name stored in CSV/pass-fail for front proxy",
    )
    p.add_argument(
        "--save-debug-image",
        type=str,
        default=_switch_default(SAVE_DEBUG_IMAGE),
        choices=["on", "off"],
        help="Save debug image overlays under <out>/debug_imgs",
    )
    p.add_argument(
        "--save-engine-region-pcd",
        type=str,
        default=_switch_default(SAVE_ENGINE_REGION_PCD),
        choices=["on", "off"],
        help="Save detected engine-left/right/front-gear region pointclouds for 3D color overlay",
    )
    p.add_argument(
        "--show-engine-regions-3d",
        type=str,
        default=_switch_default(SHOW_ENGINE_REGION_OVERLAY_3D),
        choices=["on", "off"],
        help="Show detected engine-left/right/front-gear pointcloud overlays in Open3D view",
    )
    p.add_argument(
        "--use-engine-region-ratio",
        type=str,
        default=_switch_default(USE_ENGINE_REGION_RATIO_FOR_PASSFAIL),
        choices=["on", "off"],
        help="Use warning-box coverage by detected engine-region points for pass/fail",
    )
    p.add_argument(
        "--engine-region-inside-ratio-thr",
        type=float,
        default=float(ENGINE_REGION_INSIDE_RATIO_THR),
        help="Coverage threshold (0..1) for engine-region warning pass/fail",
    )
    p.add_argument(
        "--draw-proxy-keypoints",
        type=str,
        default=_switch_default(DRAW_PROXY_KEYPOINTS_ON_DEBUG_IMAGE),
        choices=["on", "off"],
        help="Draw proxy keypoints on debug image (engine centers/front proxy)",
    )
    p.add_argument(
        "--check-bbox-coverage",
        type=str,
        default=_switch_default(CHECK_BBOX_COVERAGE),
        choices=["on", "off"],
        help="Check full-aircraft coverage of predicted aircraft bbox",
    )
    p.add_argument(
        "--visualize",
        type=str,
        default=_switch_default(VISUALIZE_PCD),
        choices=["on", "off"],
        help="Show Open3D visualization",
    )
    p.add_argument(
        "--visualize-failed-only",
        type=str,
        default=_switch_default(VISUALIZE_FAILED_SCENES_ONLY),
        choices=["on", "off"],
        help="When visualization is enabled, open only scenes with warning-result FAIL",
    )
    p.add_argument(
        "--warning-check",
        type=str,
        default=_switch_default(RUN_WARNING_CHECK),
        choices=["on", "off"],
        help="Run warning-box inside/outside checks",
    )
    p.add_argument(
        "--warning-pass-fail",
        type=str,
        default=_switch_default(RUN_WARNING_PASS_FAIL),
        choices=["on", "off"],
        help="Enable warning pass/fail scene summary",
    )
    p.add_argument(
        "--use-scene-h5-transform",
        type=str,
        default=_switch_default(USE_SCENE_H5_TRANSFORM),
        choices=["on", "off"],
        help="Use scene keypoints from H5 to place warning boxes",
    )
    p.add_argument(
        "--failed-scenes-mode",
        type=str,
        default=str(FAILED_SCENES_MODE),
        choices=["off", "list", "open"],
        help="How to handle warning FAIL scenes: off, list paths, or auto-open images",
    )
    p.add_argument(
        "--open-failed-limit",
        type=int,
        default=int(OPEN_FAILED_LIMIT),
        help="Maximum failed images to auto-open when --failed-scenes-mode=open",
    )
    p.add_argument(
        "--run-region-hypothesis",
        type=str,
        default=_switch_default(RUN_REGION_HYPOTHESIS),
        choices=["on", "off"],
        help=(
            "Run aircraft_pipeline detectors on saved region PCDs "
            "(engine_left/right and front_gear) and export hypothesis CSV."
        ),
    )
    p.add_argument(
        "--aircraft-pipeline-root",
        type=str,
        default=str(DEFAULT_AIRCRAFT_PIPELINE_ROOT),
        help="Path to aircraft_pipeline package root folder.",
    )
    p.add_argument(
        "--region-hypothesis-min-engines",
        type=int,
        default=int(REGION_HYPOTHESIS_MIN_ENGINES),
        help="Minimum detected engine regions required for PASS.",
    )
    p.add_argument(
        "--region-hypothesis-require-front-gear",
        type=str,
        default=_switch_default(REGION_HYPOTHESIS_REQUIRE_FRONT_GEAR),
        choices=["on", "off"],
        help="Require front_gear region detection as part of PASS.",
    )
    p.add_argument(
        "--region-hypothesis-python",
        type=str,
        default=str(REGION_HYPOTHESIS_PYTHON),
        help=(
            "Optional Python executable for region hypothesis subprocess "
            "(example: /home/femi/yolo_pose_dataset_creation/.venv_o3d/bin/python)."
        ),
    )
    p.add_argument(
        "--run-merger-pose",
        type=str,
        default=_switch_default(RUN_MERGER_POSE),
        choices=["on", "off"],
        help=(
            "Run aircraft_pipeline merger on scene PCDs and export pose hypothesis CSV."
        ),
    )
    p.add_argument(
        "--merger-pose-detect-all-parts",
        type=str,
        default=_switch_default(MERGER_POSE_DETECT_ALL_PARTS),
        choices=["on", "off"],
        help="Enable engine+wing+main-gear detection for merger pose stage.",
    )
    p.add_argument(
        "--merger-pose-require-final-graph",
        type=str,
        default=_switch_default(MERGER_POSE_REQUIRE_FINAL_GRAPH),
        choices=["on", "off"],
        help="Mark merger pose PASS only when final_graph_detected=yes.",
    )
    p.add_argument(
        "--merger-pose-use-region-pcd-only",
        type=str,
        default=_switch_default(MERGER_POSE_USE_REGION_PCD_ONLY),
        choices=["on", "off"],
        help=(
            "Build merger-pose input PCDs only from detected region clouds "
            "(engine_left/right + front_gear) instead of full-scene PCDs."
        ),
    )
    p.add_argument(
        "--merger-pose-python",
        type=str,
        default=str(MERGER_POSE_PYTHON),
        help=(
            "Optional Python executable for merger pose subprocess "
            "(example: /home/femi/yolo_pose_dataset_creation/.venv_o3d/bin/python)."
        ),
    )
    p.add_argument(
        "--merger-pose-visualize",
        type=str,
        default=_switch_default(MERGER_POSE_VISUALIZE),
        choices=["on", "off"],
        help="Show aircraft_pipeline final-pose Open3D windows during merger stage.",
    )
    p.add_argument(
        "--merger-pose-show-detected-parts",
        type=str,
        default=_switch_default(MERGER_POSE_SHOW_DETECTED_PARTS),
        choices=["on", "off"],
        help="In merger pose visualization, also show detected part proposals.",
    )
    p.add_argument(
        "--merger-pose-log-mode",
        type=str,
        default=str(MERGER_POSE_LOG_MODE),
        choices=["off", "minimal", "full"],
        help=(
            "Pose-stage console output level: off (silent), "
            "minimal (scene/pass/selected_components), full (all stdout/stderr)."
        ),
    )
    p.add_argument(
        "--ransac-ground-removal",
        type=str,
        default=_switch_default(RUN_RANSAC_GROUND_REMOVAL),
        choices=["on", "off"],
        help="Apply RANSAC plane-based ground removal on detected pointclouds.",
    )
    p.add_argument(
        "--ransac-ground-dist-thr",
        type=float,
        default=float(RANSAC_GROUND_DIST_THR),
        help="RANSAC inlier distance threshold for ground plane.",
    )
    p.add_argument(
        "--ransac-ground-iters",
        type=int,
        default=int(RANSAC_GROUND_MAX_ITERS),
        help="Max RANSAC iterations for ground plane estimation.",
    )
    p.add_argument(
        "--ransac-ground-min-inlier-ratio",
        type=float,
        default=float(RANSAC_GROUND_MIN_INLIER_RATIO),
        help="Minimum inlier ratio to accept a ground plane (0..1).",
    )
    p.add_argument(
        "--ransac-ground-min-abs-z",
        type=float,
        default=float(RANSAC_GROUND_MIN_ABS_Z),
        help="Require |normal_z| >= this threshold to accept a ground-like plane.",
    )
    return p.parse_args()


def _parse_det_result_best_by_class(
    result: Any, cls_id: int
) -> Optional[Tuple[Tuple[int, int, int, int], float]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return None

    cls = boxes.cls.detach().cpu().numpy().astype(int)
    conf = boxes.conf.detach().cpu().numpy().astype(float)
    xyxy = boxes.xyxy.detach().cpu().numpy().astype(float)

    best_i: Optional[int] = None
    best_conf = -1.0
    for i in range(len(cls)):
        if int(cls[i]) != int(cls_id):
            continue
        c = float(conf[i])
        if c > best_conf:
            best_conf = c
            best_i = i

    if best_i is None:
        return None

    x1, y1, x2, y2 = [int(round(v)) for v in xyxy[best_i].tolist()]
    return (x1, y1, x2, y2), float(best_conf)


def _clip_bbox_to_image(
    bb: Tuple[int, int, int, int], w: int, h: int
) -> Optional[Tuple[int, int, int, int]]:
    x1, y1, x2, y2 = [int(v) for v in bb]
    x1 = int(np.clip(x1, 0, w - 1))
    x2 = int(np.clip(x2, 0, w - 1))
    y1 = int(np.clip(y1, 0, h - 1))
    y2 = int(np.clip(y2, 0, h - 1))
    if x2 < x1 or y2 < y1:
        return None
    return x1, y1, x2, y2


def _center_px_from_bbox(bb: Tuple[int, int, int, int]) -> Tuple[int, int]:
    x1, y1, x2, y2 = [int(v) for v in bb]
    cx = int(round(0.5 * (float(x1) + float(x2))))
    cy = int(round(0.5 * (float(y1) + float(y2))))
    return cx, cy


def _normalize_engine_sides_by_image_x(
    *,
    aircraft_bb: Tuple[int, int, int, int],
    left_bb: Optional[Tuple[int, int, int, int]],
    right_bb: Optional[Tuple[int, int, int, int]],
    left_conf: float,
    right_conf: float,
) -> Tuple[
    Optional[Tuple[int, int, int, int]],
    Optional[Tuple[int, int, int, int]],
    float,
    float,
    str,
]:
    """
    Normalize engine side assignments in image space.
    - If both sides exist: enforce left.cx <= right.cx
    - If only one side exists: remap using aircraft bbox center as divider
    """
    ac_cx, _ = _center_px_from_bbox(aircraft_bb)
    reason = "none"

    lbb = left_bb
    rbb = right_bb
    lcf = float(left_conf)
    rcf = float(right_conf)

    if lbb is not None and rbb is not None:
        lcx, _ = _center_px_from_bbox(lbb)
        rcx, _ = _center_px_from_bbox(rbb)
        if lcx > rcx:
            lbb, rbb = rbb, lbb
            lcf, rcf = rcf, lcf
            reason = "swap_both_by_x_order"
        return lbb, rbb, lcf, rcf, reason

    if lbb is not None and rbb is None:
        lcx, _ = _center_px_from_bbox(lbb)
        if lcx > ac_cx:
            rbb, rcf = lbb, lcf
            lbb, lcf = None, 0.0
            reason = "move_single_left_to_right_by_aircraft_center"
        return lbb, rbb, lcf, rcf, reason

    if rbb is not None and lbb is None:
        rcx, _ = _center_px_from_bbox(rbb)
        if rcx < ac_cx:
            lbb, lcf = rbb, rcf
            rbb, rcf = None, 0.0
            reason = "move_single_right_to_left_by_aircraft_center"
        return lbb, rbb, lcf, rcf, reason

    return lbb, rbb, lcf, rcf, reason


def _extract_scene_detections_from_result(
    result: Any,
    *,
    image_w: int,
    image_h: int,
    front_class_id: int,
    force_engine_side_by_image_x: bool,
) -> Optional[Dict[str, Any]]:
    aircraft_det = _parse_det_result_best_by_class(result, CLASS_AIRCRAFT)
    if aircraft_det is None:
        return None

    aircraft_bb_raw, aircraft_conf = aircraft_det
    aircraft_bb = _clip_bbox_to_image(aircraft_bb_raw, int(image_w), int(image_h))
    if aircraft_bb is None:
        return None

    left_det = _parse_det_result_best_by_class(result, CLASS_ENGINE_LEFT)
    right_det = _parse_det_result_best_by_class(result, CLASS_ENGINE_RIGHT)
    front_det = _parse_det_result_best_by_class(result, int(front_class_id))

    eng_left_bb: Optional[Tuple[int, int, int, int]] = None
    eng_left_conf = 0.0
    if left_det is not None:
        bb_l, cf_l = left_det
        eng_left_bb = _clip_bbox_to_image(bb_l, int(image_w), int(image_h))
        eng_left_conf = float(cf_l)

    eng_right_bb: Optional[Tuple[int, int, int, int]] = None
    eng_right_conf = 0.0
    if right_det is not None:
        bb_r, cf_r = right_det
        eng_right_bb = _clip_bbox_to_image(bb_r, int(image_w), int(image_h))
        eng_right_conf = float(cf_r)

    front_bb: Optional[Tuple[int, int, int, int]] = None
    front_conf = 0.0
    if front_det is not None:
        bb_f, cf_f = front_det
        front_bb = _clip_bbox_to_image(bb_f, int(image_w), int(image_h))
        front_conf = float(cf_f)

    side_fix_reason = "none"
    if bool(force_engine_side_by_image_x):
        (
            eng_left_bb,
            eng_right_bb,
            eng_left_conf,
            eng_right_conf,
            side_fix_reason,
        ) = _normalize_engine_sides_by_image_x(
            aircraft_bb=aircraft_bb,
            left_bb=eng_left_bb,
            right_bb=eng_right_bb,
            left_conf=float(eng_left_conf),
            right_conf=float(eng_right_conf),
        )

    return {
        "aircraft_bb": aircraft_bb,
        "aircraft_conf": float(aircraft_conf),
        "eng_left_bb": eng_left_bb,
        "eng_left_conf": float(eng_left_conf),
        "eng_right_bb": eng_right_bb,
        "eng_right_conf": float(eng_right_conf),
        "front_bb": front_bb,
        "front_conf": float(front_conf),
        "side_fix_reason": str(side_fix_reason),
    }


def _extract_points_from_bbox(
    xyz_hw3: np.ndarray,
    bb: Optional[Tuple[int, int, int, int]],
    *,
    metric_margin_m: float = 0.0,
    all_finite_xyz: Optional[np.ndarray] = None,
    exclude_bbs: Optional[List[Optional[Tuple[int, int, int, int]]]] = None,
) -> np.ndarray:
    if bb is None:
        return np.empty((0, 3), dtype=np.float32)
    x1, y1, x2, y2 = [int(v) for v in bb]
    core_hw3 = xyz_hw3[y1 : y2 + 1, x1 : x2 + 1, :]
    if exclude_bbs:
        keep_mask = np.ones((y2 - y1 + 1, x2 - x1 + 1), dtype=bool)
        for ebb in exclude_bbs:
            if ebb is None:
                continue
            ex1, ey1, ex2, ey2 = [int(v) for v in ebb]
            ix1 = max(x1, ex1)
            iy1 = max(y1, ey1)
            ix2 = min(x2, ex2)
            iy2 = min(y2, ey2)
            if ix2 < ix1 or iy2 < iy1:
                continue
            keep_mask[(iy1 - y1) : (iy2 - y1 + 1), (ix1 - x1) : (ix2 - x1 + 1)] = False
        core = core_hw3[keep_mask].reshape(-1, 3)
    else:
        core = core_hw3.reshape(-1, 3)
    finite_core = np.all(np.isfinite(core), axis=1)
    core = core[finite_core]
    if core.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    margin_m = float(max(0.0, metric_margin_m))
    if margin_m <= 0.0:
        return np.asarray(core, dtype=np.float32).reshape(-1, 3)

    all_pts = all_finite_xyz
    if all_pts is None:
        all_pts = np.asarray(xyz_hw3, dtype=np.float32).reshape(-1, 3)
        finite_all = np.all(np.isfinite(all_pts), axis=1)
        all_pts = all_pts[finite_all]
    else:
        all_pts = np.asarray(all_pts, dtype=np.float32).reshape(-1, 3)
        finite_all = np.all(np.isfinite(all_pts), axis=1)
        all_pts = all_pts[finite_all]

    if all_pts.size == 0:
        return np.asarray(core, dtype=np.float32).reshape(-1, 3)

    # Strict metric expansion: keep all finite scene points whose nearest
    # bbox-seed point is within margin_m (Euclidean distance in XYZ space).
    try:
        from scipy.spatial import cKDTree  # type: ignore

        tree = cKDTree(np.asarray(core, dtype=np.float64))
        dists, _ = tree.query(
            np.asarray(all_pts, dtype=np.float64),
            k=1,
            distance_upper_bound=float(margin_m),
            workers=-1,
        )
        keep = np.isfinite(dists)
    except Exception:
        # Fallback without scipy: exact but slower blockwise nearest-distance test.
        all64 = np.asarray(all_pts, dtype=np.float64)
        core64 = np.asarray(core, dtype=np.float64)
        n_all = int(all64.shape[0])
        keep = np.zeros((n_all,), dtype=bool)
        thr2 = float(margin_m) * float(margin_m)
        q_block = 4096
        c_block = 4096
        for q0 in range(0, n_all, q_block):
            q1 = min(n_all, q0 + q_block)
            q = all64[q0:q1]
            best = np.full((q.shape[0],), np.inf, dtype=np.float64)
            for c0 in range(0, core64.shape[0], c_block):
                c1 = min(core64.shape[0], c0 + c_block)
                c = core64[c0:c1]
                d2 = np.sum((q[:, None, :] - c[None, :, :]) ** 2, axis=2)
                best = np.minimum(best, np.min(d2, axis=1))
                if np.all(best <= thr2):
                    break
            keep[q0:q1] = best <= thr2
    pts = all_pts[keep]
    if pts.size == 0:
        return np.asarray(core, dtype=np.float32).reshape(-1, 3)
    return np.asarray(pts, dtype=np.float32).reshape(-1, 3)


def _expand_region_points_vertical(
    region_pts_xyz: np.ndarray,
    *,
    all_finite_xyz: Optional[np.ndarray],
    z_expand_m: float,
) -> np.ndarray:
    """
    Expand a region cloud only in Z by +/- z_expand_m while keeping the
    original region XY bounds.
    """
    pts = np.asarray(region_pts_xyz, dtype=np.float32).reshape(-1, 3)
    if pts.shape[0] <= 0:
        return pts
    zpad = float(max(0.0, z_expand_m))
    if zpad <= 0.0:
        return pts
    if all_finite_xyz is None:
        return pts

    all_pts = np.asarray(all_finite_xyz, dtype=np.float32).reshape(-1, 3)
    finite = np.all(np.isfinite(all_pts), axis=1)
    all_pts = all_pts[finite]
    if all_pts.shape[0] <= 0:
        return pts

    mn = np.min(pts, axis=0)
    mx = np.max(pts, axis=0)
    keep = (
        (all_pts[:, 0] >= float(mn[0]))
        & (all_pts[:, 0] <= float(mx[0]))
        & (all_pts[:, 1] >= float(mn[1]))
        & (all_pts[:, 1] <= float(mx[1]))
        & (all_pts[:, 2] >= float(mn[2]) - zpad)
        & (all_pts[:, 2] <= float(mx[2]) + zpad)
    )
    out = all_pts[keep]
    if out.shape[0] <= 0:
        return pts
    return np.asarray(out, dtype=np.float32).reshape(-1, 3)


def _ground_inliers_ransac(
    pts_xyz: np.ndarray,
    *,
    dist_thr: float,
    max_iters: int,
    min_inlier_ratio: float,
    min_abs_z: float,
) -> np.ndarray:
    pts = np.asarray(pts_xyz, dtype=np.float64).reshape(-1, 3)
    n = int(pts.shape[0])
    if n < 3:
        return np.zeros((n,), dtype=bool)

    thr = float(max(1e-6, dist_thr))
    iters = int(max(10, max_iters))
    ratio = float(np.clip(min_inlier_ratio, 0.0, 1.0))
    min_nz = float(np.clip(min_abs_z, 0.0, 1.0))

    rng = np.random.default_rng(12345)
    best_mask = np.zeros((n,), dtype=bool)
    best_count = 0

    for _ in range(iters):
        try:
            i0, i1, i2 = rng.choice(n, size=3, replace=False)
        except Exception:
            break
        p0 = pts[i0]
        p1 = pts[i1]
        p2 = pts[i2]
        v1 = p1 - p0
        v2 = p2 - p0
        normal = np.cross(v1, v2)
        norm_n = float(np.linalg.norm(normal))
        if not np.isfinite(norm_n) or norm_n < 1e-10:
            continue
        normal /= norm_n
        if abs(float(normal[2])) < min_nz:
            continue
        d = -float(np.dot(normal, p0))
        dist = np.abs(pts @ normal + d)
        mask = dist <= thr
        cnt = int(np.count_nonzero(mask))
        if cnt > best_count:
            best_count = cnt
            best_mask = mask

    if best_count < max(3, int(round(ratio * float(n)))):
        return np.zeros((n,), dtype=bool)
    return best_mask


def _remove_ground_ransac(
    pts_xyz: np.ndarray,
    *,
    dist_thr: float,
    max_iters: int,
    min_inlier_ratio: float,
    min_abs_z: float,
    keep_original_if_empty: bool,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    pts = np.asarray(pts_xyz, dtype=np.float32).reshape(-1, 3)
    n_in = int(pts.shape[0])
    if n_in <= 0:
        return pts, {"input_points": 0, "ground_points": 0, "kept_points": 0, "used_fallback": 0}

    ground_mask = _ground_inliers_ransac(
        pts,
        dist_thr=float(dist_thr),
        max_iters=int(max_iters),
        min_inlier_ratio=float(min_inlier_ratio),
        min_abs_z=float(min_abs_z),
    )
    if ground_mask.shape[0] != n_in:
        ground_mask = np.zeros((n_in,), dtype=bool)

    kept = pts[~ground_mask]
    used_fallback = 0
    if kept.shape[0] <= 0 and bool(keep_original_if_empty):
        kept = pts
        used_fallback = 1

    stats = {
        "input_points": int(n_in),
        "ground_points": int(np.count_nonzero(ground_mask)),
        "kept_points": int(kept.shape[0]),
        "used_fallback": int(used_fallback),
    }
    return np.asarray(kept, dtype=np.float32).reshape(-1, 3), stats


def _draw_overlay(
    image_bgr: np.ndarray,
    aircraft_bb: Tuple[int, int, int, int],
    eng_left_bb: Optional[Tuple[int, int, int, int]],
    eng_right_bb: Optional[Tuple[int, int, int, int]],
    front_bb: Optional[Tuple[int, int, int, int]],
    kp_slots: List[Tuple[str, Tuple[int, int], float]],
) -> np.ndarray:
    out = image_bgr.copy()

    x1, y1, x2, y2 = aircraft_bb
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(out, "aircraft", (x1 + 2, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

    if eng_left_bb is not None:
        a1, b1, a2, b2 = eng_left_bb
        cv2.rectangle(out, (a1, b1), (a2, b2), (255, 200, 0), 2)
        cv2.putText(out, "engine_left", (a1 + 2, max(12, b1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1, cv2.LINE_AA)
    if eng_right_bb is not None:
        a1, b1, a2, b2 = eng_right_bb
        cv2.rectangle(out, (a1, b1), (a2, b2), (0, 200, 255), 2)
        cv2.putText(out, "engine_right", (a1 + 2, max(12, b1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA)
    if front_bb is not None:
        a1, b1, a2, b2 = front_bb
        cv2.rectangle(out, (a1, b1), (a2, b2), (255, 0, 255), 2)
        cv2.putText(out, "front_gear", (a1 + 2, max(12, b1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1, cv2.LINE_AA)

    for name, (cx, cy), conf in kp_slots:
        cv2.circle(out, (int(cx), int(cy)), 4, (0, 0, 255), -1, lineType=cv2.LINE_AA)
        cv2.putText(
            out,
            f"{name}:{float(conf):.2f}",
            (int(cx) + 4, max(12, int(cy) - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return out


def _load_scene_points_and_meta(
    grp: h5py.Group,
) -> Tuple[np.ndarray, List[str]]:
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
    return flat, cols


def _write_keypoint_conf_csv(
    rows: List[Dict[str, Any]],
    out_csv: Path,
    slot_names: List[str],
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    headers = [f"conf_{nm}" for nm in slot_names]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["unique_scene", "h5_file", "scene_name", "has_conf", "num_conf"] + headers)
        for r in rows:
            vals: List[str] = []
            conf_map: Dict[str, float] = dict(r.get("conf_map", {}))
            for nm in slot_names:
                if nm in conf_map:
                    vals.append(f"{float(conf_map[nm]):.6f}")
                else:
                    vals.append("")
            num_conf = sum(1 for nm in slot_names if nm in conf_map)
            w.writerow(
                [
                    str(r.get("unique_scene", "")),
                    str(r.get("h5_file", "")),
                    str(r.get("scene_name", "")),
                    1 if num_conf > 0 else 0,
                    num_conf,
                ]
                + vals
            )


def _write_bbox_cov_csv(rows: List[Dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
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
        for r in rows:
            rec_v = r.get("aircraft_recall", None)
            w.writerow(
                [
                    r.get("unique_scene", ""),
                    r.get("h5_file", ""),
                    r.get("scene_name", ""),
                    "" if r.get("bbox_x1", None) is None else int(r.get("bbox_x1")),
                    "" if r.get("bbox_y1", None) is None else int(r.get("bbox_y1")),
                    "" if r.get("bbox_x2", None) is None else int(r.get("bbox_x2")),
                    "" if r.get("bbox_y2", None) is None else int(r.get("bbox_y2")),
                    r.get("bbox_status", ""),
                    r.get("bbox_reason", ""),
                    "" if r.get("aircraft_px_total", None) is None else int(r.get("aircraft_px_total")),
                    "" if r.get("aircraft_px_inside", None) is None else int(r.get("aircraft_px_inside")),
                    "" if rec_v is None else f"{float(rec_v):.6f}",
                    "" if r.get("bbox_area_px", None) is None else int(r.get("bbox_area_px")),
                    f"{float(r.get('full_threshold', 0.0)):.6f}",
                ]
            )


def _collect_failed_scenes_from_passfail_csv(csv_path: Path) -> List[str]:
    if not csv_path.exists() or not csv_path.is_file():
        return []
    failed: Dict[str, bool] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            scene = str(row.get("unique_scene", "")).strip()
            if not scene:
                continue
            status = str(row.get("scene_warning_result", "")).strip().upper()
            if status == "FAIL":
                failed[scene] = True
            elif scene not in failed:
                failed[scene] = False
    out = [s for s, is_fail in failed.items() if bool(is_fail)]
    out.sort()
    return out


def _open_images_with_default_app(paths: List[Path], limit: int) -> int:
    max_n = max(0, int(limit))
    if max_n <= 0:
        return 0
    opener = shutil.which("xdg-open") or shutil.which("open")
    if opener is None:
        return 0
    opened = 0
    for p in paths[:max_n]:
        try:
            subprocess.Popen([str(opener), str(p)])
            opened += 1
        except Exception:
            continue
    return opened


def _python_can_import_module(
    python_exe: Path,
    module_name: str,
) -> bool:
    py = Path(python_exe).expanduser()
    if not py.is_absolute():
        py = (Path.cwd() / py)
    if not py.exists() or not py.is_file():
        return False
    try:
        proc = subprocess.run(
            [str(py), "-c", f"import {module_name}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return int(proc.returncode) == 0
    except Exception:
        return False


def _resolve_merger_pose_python(
    *,
    explicit_python: Optional[Path],
    region_python: Optional[Path],
) -> Path:
    candidates: List[Path] = []
    if explicit_python is not None:
        candidates.append(Path(explicit_python).expanduser())
    if region_python is not None:
        candidates.append(Path(region_python).expanduser())

    # Common local venv location for this project.
    candidates.append((Path(__file__).resolve().parent / ".venv_o3d" / "bin" / "python"))

    # Active interpreter and active venv interpreter (if any).
    candidates.append(Path(sys.executable).expanduser())
    venv = str(os.environ.get("VIRTUAL_ENV", "")).strip()
    if venv:
        candidates.append((Path(venv).expanduser() / "bin" / "python"))

    checked: List[Path] = []
    seen: set[str] = set()
    for c in candidates:
        try:
            cp = Path(c).expanduser()
            if not cp.is_absolute():
                cp = (Path.cwd() / cp)
        except Exception:
            continue
        key = str(cp)
        if key in seen:
            continue
        seen.add(key)
        checked.append(cp)
        if _python_can_import_module(cp, "open3d"):
            return cp

    checked_txt = ", ".join(str(p) for p in checked) if checked else "none"
    raise RuntimeError(
        "No Open3D-capable Python found for merger pose. "
        f"Checked: {checked_txt}. "
        "Set --merger-pose-python explicitly (example: "
        "/home/femi/yolo_pose_dataset_creation/.venv_o3d/bin/python)."
    )


_REGION_HYPOTHESIS_SUBPROCESS_CODE = r"""
import argparse
import csv
import os
import sys
from pathlib import Path
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline-root", required=True)
    ap.add_argument("--engine-region-root", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--scenes-file", required=True)
    ap.add_argument("--min-engines", type=int, required=True)
    ap.add_argument("--require-front-gear", type=int, required=True)
    ap.add_argument("--visualize", type=int, required=True)
    ap.add_argument("--vis-max-scenes", type=int, required=True)
    ap.add_argument("--center-sphere-radius", type=float, required=True)
    ap.add_argument("--split-windows", type=int, required=True)
    args = ap.parse_args()

    pipeline_root = Path(args.pipeline_root).expanduser().resolve()
    engine_region_root = Path(args.engine_region_root).expanduser().resolve()
    out_csv = Path(args.out_csv).expanduser().resolve()
    scenes_file = Path(args.scenes_file).expanduser().resolve()
    min_eng = max(1, int(args.min_engines))
    require_front = bool(int(args.require_front_gear))
    visualize = bool(int(args.visualize))
    vis_max_scenes = max(0, int(args.vis_max_scenes))
    center_sphere_radius = max(0.01, float(args.center_sphere_radius))
    split_windows = bool(int(args.split_windows))

    if not pipeline_root.exists():
        raise FileNotFoundError(f"aircraft_pipeline root not found: {pipeline_root}")
    if not scenes_file.exists():
        raise FileNotFoundError(f"scenes file not found: {scenes_file}")

    sys.path.insert(0, str(pipeline_root.parent))
    try:
        from aircraft_pipeline.engine_detector import detect_engines, EngineConfig
        from aircraft_pipeline.gear_detector import detect_gears, GearConfig
    except Exception:
        sys.path.insert(0, str(pipeline_root))
        from engine_detector import detect_engines, EngineConfig  # type: ignore
        from gear_detector import detect_gears, GearConfig  # type: ignore

    scenes = [
        s.strip() for s in scenes_file.read_text(encoding="utf-8").splitlines() if s.strip()
    ]
    eng_cfg = EngineConfig(DEBUG=False)
    try:
        gear_cfg = GearConfig(SIMPLE_NO_FILTERS=False)
    except TypeError:
        gear_cfg = GearConfig()
        try:
            setattr(gear_cfg, "SIMPLE_NO_FILTERS", False)
        except Exception:
            pass
    rows = []
    vis_rows = []

    def _normalize_clusters(raw):
        out = []
        if raw is None:
            return out
        if isinstance(raw, np.ndarray):
            raw = [raw]
        try:
            items = list(raw)
        except Exception:
            return out
        for c in items:
            arr = np.asarray(c, dtype=np.float64)
            if arr.ndim == 2 and arr.shape[0] > 0 and arr.shape[1] >= 3:
                out.append(arr[:, :3])
        return out

    def _cluster_centers(clusters):
        out = []
        for c in clusters:
            if isinstance(c, np.ndarray) and c.size > 0:
                out.append(np.asarray(np.mean(c[:, :3], axis=0), dtype=np.float64).reshape(3))
        return out

    for scene in sorted(set(scenes)):
        left_pcd = engine_region_root / "engine_left" / f"{scene}.pcd"
        right_pcd = engine_region_root / "engine_right" / f"{scene}.pcd"
        front_pcd = engine_region_root / "front_gear" / f"{scene}.pcd"

        left_count = 0
        right_count = 0
        front_nose_count = 0
        front_main_count = 0
        left_error = ""
        right_error = ""
        front_error = ""
        left_clusters = []
        right_clusters = []
        front_nose_clusters = []
        front_main_clusters = []
        left_centers = []
        right_centers = []
        front_nose_centers = []
        front_main_centers = []

        if left_pcd.exists():
            try:
                out = detect_engines(left_pcd, eng_cfg)
                if isinstance(out, tuple):
                    out = out[0]
                left_clusters = _normalize_clusters(out)
                left_count = int(len(left_clusters))
                left_centers = _cluster_centers(left_clusters)
            except Exception as e:
                left_error = f"{type(e).__name__}: {e}"
        if right_pcd.exists():
            try:
                out = detect_engines(right_pcd, eng_cfg)
                if isinstance(out, tuple):
                    out = out[0]
                right_clusters = _normalize_clusters(out)
                right_count = int(len(right_clusters))
                right_centers = _cluster_centers(right_clusters)
            except Exception as e:
                right_error = f"{type(e).__name__}: {e}"
        if front_pcd.exists():
            try:
                nose_pts, main_pts, _dbg = detect_gears(front_pcd, gear_cfg, debug_print_top=0)
            except TypeError:
                nose_pts, main_pts, _dbg = detect_gears(front_pcd, gear_cfg)
            except Exception as e:
                front_error = f"{type(e).__name__}: {e}"
                nose_pts, main_pts = [], []
            front_nose_clusters = _normalize_clusters(nose_pts)
            # Nose-gear-only mode: intentionally ignore main-gear detections.
            front_main_clusters = []
            front_nose_count = int(len(front_nose_clusters))
            front_main_count = 0
            front_nose_centers = _cluster_centers(front_nose_clusters)
            front_main_centers = []

        left_detected = left_count > 0
        right_detected = right_count > 0
        # Nose-gear-only mode: ignore main-gear detections for PASS/FAIL.
        front_detected = front_nose_count > 0
        engine_regions_detected = int(left_detected) + int(right_detected)
        if engine_regions_detected < min_eng:
            hypothesis_result = "FAIL"
            hypothesis_reason = f"engine_regions_detected<{min_eng}"
        elif require_front and not front_detected:
            hypothesis_result = "FAIL"
            hypothesis_reason = "front_gear_not_detected"
        else:
            hypothesis_result = "PASS"
            hypothesis_reason = "ok"

        rows.append(
            {
                "unique_scene": scene,
                "left_pcd_exists": int(left_pcd.exists()),
                "right_pcd_exists": int(right_pcd.exists()),
                "front_pcd_exists": int(front_pcd.exists()),
                "left_candidates": left_count,
                "right_candidates": right_count,
                "front_nose_candidates": front_nose_count,
                "front_main_candidates": front_main_count,
                "left_detected": int(left_detected),
                "right_detected": int(right_detected),
                "front_detected": int(front_detected),
                "engine_regions_detected": engine_regions_detected,
                "min_engines_required": min_eng,
                "require_front_gear": int(require_front),
                "hypothesis_result": hypothesis_result,
                "hypothesis_reason": hypothesis_reason,
                "left_error": left_error,
                "right_error": right_error,
                "front_error": front_error,
            }
        )
        vis_rows.append(
            {
                "scene": scene,
                "left_pcd": left_pcd,
                "right_pcd": right_pcd,
                "front_pcd": front_pcd,
                "left_clusters": left_clusters,
                "right_clusters": right_clusters,
                "front_nose_clusters": front_nose_clusters,
                "front_main_clusters": front_main_clusters,
                "left_centers": left_centers,
                "right_centers": right_centers,
                "front_nose_centers": front_nose_centers,
                "front_main_centers": front_main_centers,
            }
        )

    headers = [
        "unique_scene",
        "left_pcd_exists",
        "right_pcd_exists",
        "front_pcd_exists",
        "left_candidates",
        "right_candidates",
        "front_nose_candidates",
        "front_main_candidates",
        "left_detected",
        "right_detected",
        "front_detected",
        "engine_regions_detected",
        "min_engines_required",
        "require_front_gear",
        "hypothesis_result",
        "hypothesis_reason",
        "left_error",
        "right_error",
        "front_error",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for r in rows:
            w.writerow([r.get(k, "") for k in headers])

    if not visualize:
        return
    try:
        import open3d as o3d
    except Exception as e:
        print(f"[region-vis] skip (open3d unavailable): {type(e).__name__}: {e}")
        return

    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if not has_display:
        print("[region-vis] skip (no DISPLAY/WAYLAND)")
        return

    def _sphere(center_xyz, rgb):
        m = o3d.geometry.TriangleMesh.create_sphere(radius=float(center_sphere_radius))
        m.compute_vertex_normals()
        m.paint_uniform_color(rgb)
        m.translate(np.asarray(center_xyz, dtype=np.float64).reshape(3))
        return m

    def _cluster_cloud(cluster_xyz, rgb):
        arr = np.asarray(cluster_xyz, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] <= 0 or arr.shape[1] < 3:
            return None
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(arr[:, :3])
        pc.paint_uniform_color(rgb)
        return pc

    def _pcd_geom(path_like, rgb):
        if path_like is None:
            return None
        pp = Path(path_like)
        if not pp.exists():
            return None
        pc = o3d.io.read_point_cloud(str(pp))
        if len(pc.points) <= 0:
            return None
        pc.paint_uniform_color(rgb)
        return pc

    def _render_part(vr, *, part_label, pcd_key, pcd_rgb, cluster_specs, center_specs):
        geoms = []
        pc = _pcd_geom(vr.get(pcd_key), pcd_rgb)
        if pc is not None:
            geoms.append(pc)
        for key, base_rgb, use_fade in cluster_specs:
            for i, c in enumerate(vr.get(key, [])):
                rgb = base_rgb
                if use_fade:
                    if key == "left_clusters":
                        rgb = [1.00, max(0.30, 0.95 - 0.18 * (i % 4)), 0.05]
                    elif key == "right_clusters":
                        rgb = [0.05, max(0.30, 0.95 - 0.18 * (i % 4)), 1.00]
                    elif key == "front_nose_clusters":
                        rgb = [0.05, 1.00, max(0.20, 0.85 - 0.15 * (i % 4))]
                    elif key == "front_main_clusters":
                        rgb = [1.00, max(0.20, 0.80 - 0.15 * (i % 4)), 0.10]
                cc = _cluster_cloud(c, rgb)
                if cc is not None:
                    geoms.append(cc)
        for key, rgb in center_specs:
            for c in vr.get(key, []):
                geoms.append(_sphere(c, rgb))
        if not geoms:
            return False
        o3d.visualization.draw_geometries(
            geoms,
            window_name=(
                "REGION-HYPOTHESIS "
                f"[{part_label}] (clusters+centers) — {vr.get('scene', 'scene')}"
            ),
        )
        return True

    shown = 0
    for vr in vis_rows:
        if vis_max_scenes > 0 and shown >= vis_max_scenes:
            break
        scene_shown = False
        if split_windows:
            if _render_part(
                vr,
                part_label="front_gear(nose)",
                pcd_key="front_pcd",
                pcd_rgb=[0.25, 0.00, 0.25],
                cluster_specs=[
                    ("front_nose_clusters", [0.05, 1.00, 0.80], True),
                    ("front_main_clusters", [1.00, 0.35, 0.10], True),
                ],
                center_specs=[
                    ("front_nose_centers", [0.00, 1.00, 0.00]),
                    ("front_main_centers", [1.00, 0.20, 0.20]),
                ],
            ):
                scene_shown = True
            if _render_part(
                vr,
                part_label="engine_left",
                pcd_key="left_pcd",
                pcd_rgb=[0.35, 0.20, 0.00],
                cluster_specs=[("left_clusters", [1.00, 0.90, 0.10], True)],
                center_specs=[("left_centers", [1.00, 0.90, 0.10])],
            ):
                scene_shown = True
            if _render_part(
                vr,
                part_label="engine_right",
                pcd_key="right_pcd",
                pcd_rgb=[0.00, 0.25, 0.35],
                cluster_specs=[("right_clusters", [0.20, 0.95, 1.00], True)],
                center_specs=[("right_centers", [0.20, 0.95, 1.00])],
            ):
                scene_shown = True
        else:
            geoms = []
            for pkey, col in [
                ("left_pcd", [0.35, 0.20, 0.00]),
                ("right_pcd", [0.00, 0.25, 0.35]),
                ("front_pcd", [0.25, 0.00, 0.25]),
            ]:
                pc = _pcd_geom(vr.get(pkey), col)
                if pc is not None:
                    geoms.append(pc)
            for i, c in enumerate(vr.get("left_clusters", [])):
                cc = _cluster_cloud(c, [1.00, max(0.30, 0.95 - 0.18 * (i % 4)), 0.05])
                if cc is not None:
                    geoms.append(cc)
            for i, c in enumerate(vr.get("right_clusters", [])):
                cc = _cluster_cloud(c, [0.05, max(0.30, 0.95 - 0.18 * (i % 4)), 1.00])
                if cc is not None:
                    geoms.append(cc)
            for i, c in enumerate(vr.get("front_nose_clusters", [])):
                cc = _cluster_cloud(c, [0.05, 1.00, max(0.20, 0.85 - 0.15 * (i % 4))])
                if cc is not None:
                    geoms.append(cc)
            for i, c in enumerate(vr.get("front_main_clusters", [])):
                cc = _cluster_cloud(c, [1.00, max(0.20, 0.80 - 0.15 * (i % 4)), 0.10])
                if cc is not None:
                    geoms.append(cc)
            for c in vr.get("left_centers", []):
                geoms.append(_sphere(c, [1.00, 0.90, 0.10]))
            for c in vr.get("right_centers", []):
                geoms.append(_sphere(c, [0.20, 0.95, 1.00]))
            for c in vr.get("front_nose_centers", []):
                geoms.append(_sphere(c, [0.00, 1.00, 0.00]))
            for c in vr.get("front_main_centers", []):
                geoms.append(_sphere(c, [1.00, 0.20, 0.20]))
            if geoms:
                o3d.visualization.draw_geometries(
                    geoms,
                    window_name=f"REGION-HYPOTHESIS (clusters+centers) — {vr.get('scene', 'scene')}",
                )
                scene_shown = True
        if scene_shown:
            shown += 1

if __name__ == "__main__":
    main()
"""


_MERGER_POSE_SUBPROCESS_CODE = r"""
import argparse
import csv
import importlib
import os
import re
import sys
from collections import Counter
from pathlib import Path

def _resolve_profile_fuzzy(raw, resolver):
    if not raw or resolver is None:
        return None
    base = str(raw).strip("_ ").strip()
    if not base:
        return None

    candidates = []
    candidates.append(base)
    candidates.append(base.replace("_", "-"))
    candidates.append(base.replace("_", " "))

    if re.match(r"^[bB]\d", base):
        b = base[1:]
        candidates.append(b)
        candidates.append(b.replace("_", "-"))
        candidates.append(b.replace("_", " "))

    seed = base.replace("_", " ")
    candidates.append(re.sub(r"(\d)([A-Za-z])", r"\1 \2", seed))
    candidates.append(re.sub(r"([A-Za-z])(\d)", r"\1 \2", seed))
    candidates.append(re.sub(r"([A-Za-z]+)(\d+)$", r"\1 \2", seed))

    seen = set()
    for c in candidates:
        c = str(c).strip()
        if not c or c in seen:
            continue
        seen.add(c)
        try:
            resolved = resolver(c)
        except Exception:
            resolved = None
        if resolved:
            return str(resolved)
    return None

def _infer_profile_from_scene_stem(scene_stem, resolver):
    if not scene_stem or resolver is None:
        return None
    stem = str(scene_stem).strip()
    if not stem:
        return None

    m = re.match(r"^movement_(.+?)(?:__|$)", stem, flags=re.IGNORECASE)
    if m:
        token = str(m.group(1)).strip("_")
        parts = [p for p in token.split("_") if p]
        while parts and parts[-1].lower() in {"split", "fix"}:
            parts.pop()
        if parts:
            p = _resolve_profile_fuzzy("_".join(parts), resolver)
            if p:
                return p

    p = _resolve_profile_fuzzy(stem, resolver)
    if p:
        return p

    toks = [t for t in re.split(r"[^A-Za-z0-9]+", stem) if t]
    seen = set()
    for i in range(len(toks)):
        for w in (3, 2, 1):
            j = i + w
            if j > len(toks):
                continue
            frag = "_".join(toks[i:j]).strip("_")
            if not frag or frag in seen:
                continue
            seen.add(frag)
            if not any(ch.isdigit() for ch in frag):
                continue
            p = _resolve_profile_fuzzy(frag, resolver)
            if p:
                return p
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline-root", required=True)
    ap.add_argument("--pcd-root", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--detect-all-parts", type=int, required=True)
    ap.add_argument("--require-final-graph", type=int, required=True)
    ap.add_argument("--visualize", type=int, required=True)
    ap.add_argument("--show-detected-parts", type=int, required=True)
    ap.add_argument("--log-mode", default="full")
    args = ap.parse_args()

    pipeline_root = Path(args.pipeline_root).expanduser().resolve()
    pcd_root = Path(args.pcd_root).expanduser().resolve()
    out_csv = Path(args.out_csv).expanduser().resolve()
    detect_all = bool(int(args.detect_all_parts))
    require_final = bool(int(args.require_final_graph))
    visualize = bool(int(args.visualize))
    show_detected_parts = bool(int(args.show_detected_parts))
    log_mode = str(getattr(args, "log_mode", "full") or "").strip().lower()
    if log_mode not in {"off", "minimal", "full"}:
        log_mode = "full"

    import builtins
    _ORIG_PRINT = builtins.print

    def _emit(msg):
        _ORIG_PRINT(str(msg), flush=True)

    def _log_full(msg):
        if log_mode == "full":
            _emit(msg)

    if log_mode in {"off", "minimal"}:
        builtins.print = lambda *a, **k: None

    if not pipeline_root.exists() or not pipeline_root.is_dir():
        raise FileNotFoundError(f"aircraft_pipeline root not found: {pipeline_root}")
    if not pcd_root.exists() or not pcd_root.is_dir():
        raise FileNotFoundError(f"pcd_root not found: {pcd_root}")

    sys.path.insert(0, str(pipeline_root.parent))
    pkg_name = str(pipeline_root.name).strip()
    ModularConfig = None
    run_pipeline = None
    active_pipe_mod = None
    resolve_profile_name = None
    tried = []

    # Prefer package imports so relative imports inside pipeline modules work.
    for cand_pkg in ["aircraft_pipeline", pkg_name]:
        if not cand_pkg:
            continue
        try:
            cfg_mod = importlib.import_module(f"{cand_pkg}.config")
            pipe_mod = importlib.import_module(f"{cand_pkg}.pipeline")
            prof_mod = importlib.import_module(f"{cand_pkg}.aircraft_profiles")
            ModularConfig = getattr(cfg_mod, "ModularConfig")
            run_pipeline = getattr(pipe_mod, "run_pipeline")
            resolve_profile_name = getattr(prof_mod, "resolve_profile_name", None)
            active_pipe_mod = pipe_mod
            break
        except Exception as e:
            tried.append(f"{cand_pkg}:{type(e).__name__}:{e}")

    if ModularConfig is None or run_pipeline is None:
        # Last fallback for flat module layout (no package).
        sys.path.insert(0, str(pipeline_root))
        try:
            from config import ModularConfig  # type: ignore
            import pipeline as pipe_mod  # type: ignore
            from pipeline import run_pipeline  # type: ignore
            from aircraft_profiles import resolve_profile_name  # type: ignore
            active_pipe_mod = pipe_mod
        except Exception as e:
            tried_txt = " | ".join(tried) if tried else "none"
            raise RuntimeError(
                f"Unable to import merger pipeline modules. tried={tried_txt}"
            ) from e

    # Force strict gear-mode in merger stage so engine clusters are not
    # promoted to nose candidates when front-gear evidence is weak/missing.
    try:
        if active_pipe_mod is not None and hasattr(active_pipe_mod, "core"):
            _core = getattr(active_pipe_mod, "core")
            _orig_gear_cfg = getattr(_core, "GearConfig", None)
            if _orig_gear_cfg is not None:
                def _strict_gear_cfg_factory(**kwargs):
                    kw = dict(kwargs)
                    kw.setdefault("SIMPLE_NO_FILTERS", False)
                    return _orig_gear_cfg(**kw)
                _core.GearConfig = _strict_gear_cfg_factory
                _log_full("[MERGER-POSE] forced GearConfig SIMPLE_NO_FILTERS=False")
    except Exception as _e:
        _log_full(f"[MERGER-POSE] strict GearConfig patch skipped: {type(_e).__name__}: {_e}")

    cfg = ModularConfig()
    cfg.pcd_folder = pcd_root
    cfg.front_gear_region_folder = pcd_root / "front_gear"
    cfg.recursive = False
    cfg.max_files = None
    # Keep a visual window even when final graph is missing (FAIL scenes).
    cfg.show_overlay = bool(visualize)
    cfg.show_trip_hyps = False
    cfg.show_quad_hyps = False
    cfg.show_final_graph = bool(visualize)
    cfg.show_defined_graph_overlay = False
    cfg.show_detected_parts_in_final_graph = bool(show_detected_parts)
    cfg.show_candidate_ids = False
    cfg.show_warning_boxes = False
    cfg.show_warning_derived_graph = bool(visualize)
    cfg.save_csv = False
    cfg.run_engine_detector = True
    cfg.run_gear_detector = True
    # Allow final graph recovery from NE pair(s) when triplets/quads are unavailable.
    cfg.final_ne_pair_pose_fallback_on_missing = True
    if detect_all:
        cfg.run_wing_detector = True
        cfg.run_main_gear_detector = True
    else:
        cfg.run_wing_detector = False
        cfg.run_main_gear_detector = False

    # Infer aircraft profile from scene stems so triplet gates (e.g., NEE angle)
    # use model-specific ranges instead of merger_box defaults.
    pcd_files = sorted([p for p in pcd_root.glob("*.pcd") if p.is_file()])
    if not pcd_files:
        pcd_files = sorted([p for p in pcd_root.rglob("*.pcd") if p.is_file()])
    inferred_profiles = []
    for pp in pcd_files:
        prof = _infer_profile_from_scene_stem(pp.stem, resolve_profile_name)
        if prof:
            inferred_profiles.append(str(prof))
    if inferred_profiles:
        counts = Counter(inferred_profiles)
        best, n_best = counts.most_common(1)[0]
        cfg.aircraft_profile = str(best)
        _log_full(
            f"[MERGER-PROFILE] inferred profile={best} "
            f"(matches={int(n_best)}/{int(len(pcd_files))})"
        )
        if len(counts) > 1:
            _log_full(
                f"[MERGER-PROFILE] mixed inferred profiles={dict(counts)}; "
                f"using most-common={best}"
            )
    else:
        _log_full(
            "[MERGER-PROFILE] no profile inferred from scene stems; "
            f"using defaults NEE=[{float(getattr(cfg, 'nee_ang_min_deg', 45.0)):.1f},"
            f"{float(getattr(cfg, 'nee_ang_max_deg', 95.0)):.1f}]"
        )

    if cfg.show_final_graph:
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        if not has_display:
            cfg.show_final_graph = False
            cfg.show_detected_parts_in_final_graph = False
            _log_full("[MERGER-POSE] no DISPLAY/WAYLAND; visualization disabled")

    result = run_pipeline(cfg)
    recs = result.get("records", []) if isinstance(result, dict) else []

    full_fields = [
        "bag_name",
        "h5_file",
        "pcd_file",
        "scene_name",
        "scene_runtime_s",
        "aircraft_profile",
        "aircraft_model_raw",
        "final_graph_detected",
        "whole_h5_pass",
        "base_kind",
        "base_ids",
        "edges_count",
        "selected_components",
        "nose_proposal_count",
        "nose_idx",
        "nose_detected",
        "nose_pass",
        "main_gear_1_idx",
        "main_gear_1_detected",
        "main_gear_1_pass",
        "main_gear_2_idx",
        "main_gear_2_detected",
        "main_gear_2_pass",
        "engine_proposal_count",
        "engine_1_idx",
        "engine_1_detected",
        "engine_1_pass",
        "engine_2_idx",
        "engine_2_detected",
        "engine_2_pass",
        "wing_1_idx",
        "wing_1_detected",
        "wing_1_pass",
        "wing_2_idx",
        "wing_2_detected",
        "wing_2_pass",
        "warning_check_ok",
        "warning_components",
        "warning_pass_count",
        "warning_fail_count",
        "warning_based_source",
        "warning_based_graph_detected",
        "warning_based_base_kind",
        "warning_based_base_ids",
        "warning_based_selected_components",
        "warning_based_total_components",
        "warning_based_not_enough_true_positives",
        "warning_proximity_reason",
        "warning_pose_rot_err_enabled",
        "warning_pose_rot_err_ok",
        "warning_pose_yaw_ref_deg",
        "warning_pose_yaw_pred_deg",
        "warning_pose_dyaw_deg",
        "warning_pose_rot_angle_deg",
    ]
    headers = list(full_fields)
    for k in ["unique_scene", "pose_result", "pose_reason"]:
        if k not in headers:
            headers.append(k)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        w.writeheader()
        for rec in recs:
            if not isinstance(rec, dict):
                continue
            pcd_file = str(rec.get("pcd_file", "") or "")
            scene = str(rec.get("scene_name", "") or "").strip()
            if not scene:
                scene = Path(pcd_file).stem if pcd_file else ""
            unique_scene = str(rec.get("unique_scene", "") or "").strip()
            if not unique_scene:
                if scene and "__scene_" in scene:
                    unique_scene = scene
                else:
                    bag_guess = str(rec.get("bag_name", "") or "").strip()
                    if bag_guess and scene:
                        if scene.startswith("scene_"):
                            unique_scene = f"{bag_guess}__{scene}"
                        else:
                            unique_scene = scene
            final_raw = str(rec.get("final_graph_detected", "") or "").strip().lower()
            final_ok = final_raw in {"yes", "true", "1", "detected"}
            if require_final and not final_ok:
                pose_result = "FAIL"
                pose_reason = "no_final_graph"
            else:
                pose_result = "PASS"
                pose_reason = "ok"
            selected_components = str(rec.get("selected_components", ""))
            if log_mode == "full" and visualize:
                _emit(
                    f"[MERGER-POSE] scene={scene or '?'} "
                    f"final_graph_detected={final_ok} pose_result={pose_result} "
                    f"selected_components={selected_components or '-'}"
                )
            elif log_mode == "minimal":
                _emit(
                    f"[MERGER-POSE] scene={scene or '?'} "
                    f"pose_result={pose_result} selected_components={selected_components or '-'}"
                )
            out_row = {k: rec.get(k, "") for k in headers}
            out_row["unique_scene"] = str(unique_scene or scene)
            out_row["pose_result"] = str(pose_result)
            out_row["pose_reason"] = str(pose_reason)
            out_row["pcd_file"] = str(pcd_file)
            if not str(out_row.get("scene_name", "")).strip():
                if scene and "__scene_" in scene and "__" in scene:
                    out_row["scene_name"] = str(scene.split("__", 1)[1])
                else:
                    out_row["scene_name"] = str(scene)
            if not str(out_row.get("bag_name", "")).strip():
                uq = str(out_row.get("unique_scene", "")).strip()
                if "__scene_" in uq:
                    out_row["bag_name"] = uq.rsplit("__scene_", 1)[0]
            if not str(out_row.get("h5_file", "")).strip():
                bag_name = str(out_row.get("bag_name", "")).strip()
                if bag_name:
                    out_row["h5_file"] = f"{bag_name}.h5"
            if not str(out_row.get("final_graph_detected", "")).strip():
                out_row["final_graph_detected"] = ("detected" if final_ok else "not_detected")
            whole = str(out_row.get("whole_h5_pass", "")).strip().lower()
            if not whole:
                if pose_result == "PASS":
                    whole = "pass"
                else:
                    whole = ("not_detected" if not final_ok else "fail")
                out_row["whole_h5_pass"] = whole
            w.writerow(out_row)

if __name__ == "__main__":
    main()
"""


def _load_aircraft_pipeline_region_detectors(pipeline_root: Path):
    root = Path(pipeline_root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"aircraft_pipeline root not found: {root}")

    pkg_parent = root.parent
    if str(pkg_parent) not in sys.path:
        sys.path.insert(0, str(pkg_parent))

    try:
        from aircraft_pipeline.engine_detector import EngineConfig, detect_engines
        from aircraft_pipeline.gear_detector import GearConfig, detect_gears
        return detect_engines, EngineConfig, detect_gears, GearConfig
    except Exception:
        # Fallback when direct package import is not available.
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from engine_detector import EngineConfig, detect_engines  # type: ignore
        from gear_detector import GearConfig, detect_gears  # type: ignore
        return detect_engines, EngineConfig, detect_gears, GearConfig


def _visualize_engine_detector_region_outputs(
    *,
    unique_scenes: List[str],
    engine_region_root: Path,
    aircraft_pipeline_root: Path,
    max_scenes: int,
    center_sphere_radius: float,
) -> None:
    try:
        detect_engines, EngineConfig, _detect_gears, _GearConfig = (
            _load_aircraft_pipeline_region_detectors(aircraft_pipeline_root)
        )
    except Exception as e:
        print(f"[engine-det-vis] skip (detector import failed): {type(e).__name__}: {e}")
        return

    eng_cfg = EngineConfig(DEBUG=False)
    region_root = Path(engine_region_root).expanduser().resolve()
    max_n = max(0, int(max_scenes))
    shown = 0

    def _normalize_clusters(raw: Any) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        if raw is None:
            return out
        if isinstance(raw, np.ndarray):
            raw = [raw]
        try:
            items = list(raw)
        except Exception:
            return out
        for c in items:
            arr = np.asarray(c, dtype=np.float64)
            if arr.ndim == 2 and arr.shape[0] > 0 and arr.shape[1] >= 3:
                out.append(arr[:, :3])
        return out

    def _cluster_centers(clusters: List[np.ndarray]) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        for c in clusters:
            if c.size > 0:
                out.append(np.asarray(np.mean(c[:, :3], axis=0), dtype=np.float64).reshape(3))
        return out

    def _fmt_centers(centers: List[np.ndarray], max_show: int = 3) -> str:
        if not centers:
            return "-"
        txt: List[str] = []
        n_show = max(1, int(max_show))
        for c in centers[:n_show]:
            v = np.asarray(c, dtype=np.float64).reshape(3)
            txt.append(f"({v[0]:.3f},{v[1]:.3f},{v[2]:.3f})")
        if len(centers) > n_show:
            txt.append(f"... +{len(centers) - n_show}")
        return " ".join(txt)

    try:
        import open3d as o3d
    except Exception as e:
        print(f"[engine-det-vis] skip (open3d unavailable): {type(e).__name__}: {e}")
        return

    has_display = bool(
        str(os.environ.get("DISPLAY", "")).strip()
        or str(os.environ.get("WAYLAND_DISPLAY", "")).strip()
    )
    if not has_display:
        print("[engine-det-vis] skip (no DISPLAY/WAYLAND)")
        return

    def _cluster_cloud(cluster_xyz: np.ndarray, rgb: List[float]):
        arr = np.asarray(cluster_xyz, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] <= 0 or arr.shape[1] < 3:
            return None
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(arr[:, :3])
        pc.paint_uniform_color(rgb)
        return pc

    def _sphere(center_xyz: np.ndarray, rgb: List[float]):
        m = o3d.geometry.TriangleMesh.create_sphere(radius=float(center_sphere_radius))
        m.compute_vertex_normals()
        m.paint_uniform_color(rgb)
        m.translate(np.asarray(center_xyz, dtype=np.float64).reshape(3))
        return m

    for scene in sorted(set(unique_scenes)):
        if max_n > 0 and shown >= max_n:
            break
        scene_has_window = False
        for part_name, base_col, center_col in [
            ("engine_left", [0.35, 0.20, 0.00], [1.00, 0.90, 0.10]),
            ("engine_right", [0.00, 0.25, 0.35], [0.20, 0.95, 1.00]),
        ]:
            pcd_path = region_root / part_name / f"{scene}.pcd"
            if not pcd_path.exists():
                print(f"[engine-det-vis] {scene} {part_name}: missing pcd")
                continue

            try:
                det_out = detect_engines(pcd_path, eng_cfg)
                if isinstance(det_out, tuple):
                    det_out = det_out[0]
                clusters = _normalize_clusters(det_out)
            except Exception as e:
                print(
                    f"[engine-det-vis] {scene} {part_name}: detector error "
                    f"{type(e).__name__}: {e}"
                )
                continue

            centers = _cluster_centers(clusters)
            print(
                f"[engine-det-vis] {scene} {part_name}: "
                f"clusters={len(clusters)} centers={_fmt_centers(centers)}"
            )

            pcd = o3d.io.read_point_cloud(str(pcd_path))
            geoms: List[Any] = []
            if len(pcd.points) > 0:
                pcd.paint_uniform_color(base_col)
                geoms.append(pcd)
            for i, c in enumerate(clusters):
                cc = _cluster_cloud(c, [1.00, max(0.30, 0.95 - 0.18 * (i % 4)), 0.05] if part_name == "engine_left" else [0.05, max(0.30, 0.95 - 0.18 * (i % 4)), 1.00])
                if cc is not None:
                    geoms.append(cc)
            for c in centers:
                geoms.append(_sphere(c, center_col))
            if not geoms:
                continue
            o3d.visualization.draw_geometries(
                geoms,
                window_name=f"ENGINE-DETECTOR [{part_name}] (clusters+centers) — {scene}",
            )
            scene_has_window = True
        if scene_has_window:
            shown += 1


def _visualize_region_hypothesis_rows(
    rows: List[Dict[str, Any]],
    *,
    max_scenes: int,
    center_sphere_radius: float,
    split_windows: bool,
) -> None:
    try:
        import open3d as o3d
    except Exception as e:
        print(f"[region-vis] skip (open3d unavailable): {type(e).__name__}: {e}")
        return

    has_display = bool(
        str(os.environ.get("DISPLAY", "")).strip()
        or str(os.environ.get("WAYLAND_DISPLAY", "")).strip()
    )
    if not has_display:
        print("[region-vis] skip (no DISPLAY/WAYLAND)")
        return

    def _sphere(center_xyz: np.ndarray, rgb: List[float]):
        m = o3d.geometry.TriangleMesh.create_sphere(radius=float(center_sphere_radius))
        m.compute_vertex_normals()
        m.paint_uniform_color(rgb)
        m.translate(np.asarray(center_xyz, dtype=np.float64).reshape(3))
        return m

    def _cluster_cloud(cluster_xyz: Any, rgb: List[float]):
        arr = np.asarray(cluster_xyz, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] <= 0 or arr.shape[1] < 3:
            return None
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(arr[:, :3])
        pc.paint_uniform_color(rgb)
        return pc

    def _pcd_geom(path_like: Any, rgb: List[float]):
        if path_like is None:
            return None
        pp = Path(path_like)
        if not pp.exists():
            return None
        pc = o3d.io.read_point_cloud(str(pp))
        if len(pc.points) <= 0:
            return None
        pc.paint_uniform_color(rgb)
        return pc

    def _draw_scene_combined(vr: Dict[str, Any]) -> bool:
        geoms = []
        for pkey, col in [
            ("left_pcd", [0.35, 0.20, 0.00]),
            ("right_pcd", [0.00, 0.25, 0.35]),
            ("front_pcd", [0.25, 0.00, 0.25]),
        ]:
            pc = _pcd_geom(vr.get(pkey, None), col)
            if pc is not None:
                geoms.append(pc)
        for i, c in enumerate(vr.get("left_clusters", [])):
            cc = _cluster_cloud(c, [1.00, max(0.30, 0.95 - 0.18 * (i % 4)), 0.05])
            if cc is not None:
                geoms.append(cc)
        for i, c in enumerate(vr.get("right_clusters", [])):
            cc = _cluster_cloud(c, [0.05, max(0.30, 0.95 - 0.18 * (i % 4)), 1.00])
            if cc is not None:
                geoms.append(cc)
        for i, c in enumerate(vr.get("front_nose_clusters", [])):
            cc = _cluster_cloud(c, [0.05, 1.00, max(0.20, 0.85 - 0.15 * (i % 4))])
            if cc is not None:
                geoms.append(cc)
        for i, c in enumerate(vr.get("front_main_clusters", [])):
            cc = _cluster_cloud(c, [1.00, max(0.20, 0.80 - 0.15 * (i % 4)), 0.10])
            if cc is not None:
                geoms.append(cc)
        for c in vr.get("left_centers", []):
            geoms.append(_sphere(np.asarray(c, dtype=np.float64), [1.00, 0.90, 0.10]))
        for c in vr.get("right_centers", []):
            geoms.append(_sphere(np.asarray(c, dtype=np.float64), [0.20, 0.95, 1.00]))
        for c in vr.get("front_nose_centers", []):
            geoms.append(_sphere(np.asarray(c, dtype=np.float64), [0.00, 1.00, 0.00]))
        for c in vr.get("front_main_centers", []):
            geoms.append(_sphere(np.asarray(c, dtype=np.float64), [1.00, 0.20, 0.20]))
        if not geoms:
            return False
        o3d.visualization.draw_geometries(
            geoms,
            window_name=f"REGION-HYPOTHESIS (clusters+centers) — {vr.get('scene', 'scene')}",
        )
        return True

    def _draw_scene_part(
        vr: Dict[str, Any],
        *,
        part_label: str,
        pcd_key: str,
        pcd_rgb: List[float],
        cluster_mode: str,
        center_key: str,
        center_rgb: List[float],
    ) -> bool:
        geoms = []
        pc = _pcd_geom(vr.get(pcd_key, None), pcd_rgb)
        if pc is not None:
            geoms.append(pc)
        cluster_key = f"{cluster_mode}_clusters"
        for i, c in enumerate(vr.get(cluster_key, [])):
            if cluster_mode == "left":
                rgb = [1.00, max(0.30, 0.95 - 0.18 * (i % 4)), 0.05]
            elif cluster_mode == "right":
                rgb = [0.05, max(0.30, 0.95 - 0.18 * (i % 4)), 1.00]
            elif cluster_mode == "front_nose":
                rgb = [0.05, 1.00, max(0.20, 0.85 - 0.15 * (i % 4))]
            else:
                rgb = [1.00, max(0.20, 0.80 - 0.15 * (i % 4)), 0.10]
            cc = _cluster_cloud(c, rgb)
            if cc is not None:
                geoms.append(cc)
        for c in vr.get(center_key, []):
            geoms.append(_sphere(np.asarray(c, dtype=np.float64), center_rgb))
        if not geoms:
            return False
        o3d.visualization.draw_geometries(
            geoms,
            window_name=(
                "REGION-HYPOTHESIS "
                f"[{part_label}] (clusters+centers) — {vr.get('scene', 'scene')}"
            ),
        )
        return True

    shown = 0
    max_n = max(0, int(max_scenes))
    for vr in rows:
        if max_n > 0 and shown >= max_n:
            break
        scene_shown = False
        if bool(split_windows):
            if _draw_scene_part(
                vr,
                part_label="front_gear(nose)",
                pcd_key="front_pcd",
                pcd_rgb=[0.25, 0.00, 0.25],
                cluster_mode="front_nose",
                center_key="front_nose_centers",
                center_rgb=[0.00, 1.00, 0.00],
            ):
                scene_shown = True
            if _draw_scene_part(
                vr,
                part_label="engine_left",
                pcd_key="left_pcd",
                pcd_rgb=[0.35, 0.20, 0.00],
                cluster_mode="left",
                center_key="left_centers",
                center_rgb=[1.00, 0.90, 0.10],
            ):
                scene_shown = True
            if _draw_scene_part(
                vr,
                part_label="engine_right",
                pcd_key="right_pcd",
                pcd_rgb=[0.00, 0.25, 0.35],
                cluster_mode="right",
                center_key="right_centers",
                center_rgb=[0.20, 0.95, 1.00],
            ):
                scene_shown = True
        else:
            scene_shown = _draw_scene_combined(vr)
        if scene_shown:
            shown += 1


def _run_region_hypothesis(
    *,
    unique_scenes: List[str],
    engine_region_root: Path,
    out_root: Path,
    aircraft_pipeline_root: Path,
    min_engines: int,
    require_front_gear: bool,
    visualize_outputs: bool,
    vis_max_scenes: int,
    center_sphere_radius: float,
    split_windows: bool,
) -> Tuple[Path, int, int]:
    try:
        detect_engines, EngineConfig, detect_gears, GearConfig = (
            _load_aircraft_pipeline_region_detectors(aircraft_pipeline_root)
        )
    except ModuleNotFoundError as e:
        if "open3d" in str(e):
            raise RuntimeError(
                "aircraft_pipeline region hypothesis needs open3d in the active env. "
                "Run this script in an Open3D-enabled venv (for example .venv_o3d)."
            ) from e
        raise
    min_eng = max(1, int(min_engines))
    require_front = bool(require_front_gear)

    eng_cfg = EngineConfig(DEBUG=False)
    try:
        gear_cfg = GearConfig(SIMPLE_NO_FILTERS=False)
    except TypeError:
        gear_cfg = GearConfig()
        try:
            setattr(gear_cfg, "SIMPLE_NO_FILTERS", False)
        except Exception:
            pass
    rows: List[Dict[str, Any]] = []
    vis_rows: List[Dict[str, Any]] = []

    def _fmt_centers(centers: List[np.ndarray], max_show: int = 3) -> str:
        if not centers:
            return "-"
        out: List[str] = []
        for c in centers[: max(1, int(max_show))]:
            v = np.asarray(c, dtype=np.float64).reshape(3)
            out.append(f"({v[0]:.3f},{v[1]:.3f},{v[2]:.3f})")
        if len(centers) > max_show:
            out.append(f"... +{len(centers) - max_show}")
        return " ".join(out)

    def _normalize_clusters(raw: Any) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        if raw is None:
            return out
        if isinstance(raw, np.ndarray):
            raw = [raw]
        try:
            items = list(raw)
        except Exception:
            return out
        for c in items:
            arr = np.asarray(c, dtype=np.float64)
            if arr.ndim == 2 and arr.shape[0] > 0 and arr.shape[1] >= 3:
                out.append(arr[:, :3])
        return out

    def _cluster_centers(clusters: List[np.ndarray]) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        for c in clusters:
            if c.size > 0:
                out.append(np.asarray(np.mean(c[:, :3], axis=0), dtype=np.float64).reshape(3))
        return out

    for scene in sorted(set(unique_scenes)):
        left_pcd = engine_region_root / "engine_left" / f"{scene}.pcd"
        right_pcd = engine_region_root / "engine_right" / f"{scene}.pcd"
        front_pcd = engine_region_root / "front_gear" / f"{scene}.pcd"

        left_count = 0
        right_count = 0
        front_nose_count = 0
        front_main_count = 0
        left_error = ""
        right_error = ""
        front_error = ""
        left_clusters: List[np.ndarray] = []
        right_clusters: List[np.ndarray] = []
        front_nose_clusters: List[np.ndarray] = []
        front_main_clusters: List[np.ndarray] = []
        left_centers: List[np.ndarray] = []
        right_centers: List[np.ndarray] = []
        front_nose_centers: List[np.ndarray] = []
        front_main_centers: List[np.ndarray] = []

        if left_pcd.exists():
            try:
                left_out = detect_engines(left_pcd, eng_cfg)
                if isinstance(left_out, tuple):
                    left_out = left_out[0]
                left_clusters = _normalize_clusters(left_out)
                left_count = int(len(left_clusters))
                left_centers = _cluster_centers(left_clusters)
            except Exception as e:
                left_error = f"{type(e).__name__}: {e}"
        if right_pcd.exists():
            try:
                right_out = detect_engines(right_pcd, eng_cfg)
                if isinstance(right_out, tuple):
                    right_out = right_out[0]
                right_clusters = _normalize_clusters(right_out)
                right_count = int(len(right_clusters))
                right_centers = _cluster_centers(right_clusters)
            except Exception as e:
                right_error = f"{type(e).__name__}: {e}"
        if front_pcd.exists():
            try:
                nose_pts, main_pts, _dbg = detect_gears(front_pcd, gear_cfg, debug_print_top=0)
            except TypeError:
                nose_pts, main_pts, _dbg = detect_gears(front_pcd, gear_cfg)
            except Exception as e:
                front_error = f"{type(e).__name__}: {e}"
                nose_pts, main_pts = [], []
            front_nose_clusters = _normalize_clusters(nose_pts)
            # Nose-gear-only mode: intentionally ignore main-gear detections.
            front_main_clusters = []
            front_nose_count = int(len(front_nose_clusters))
            front_main_count = 0
            front_nose_centers = _cluster_centers(front_nose_clusters)
            front_main_centers = []

        left_detected = left_count > 0
        right_detected = right_count > 0
        # Nose-gear-only mode: ignore main-gear detections for PASS/FAIL.
        front_detected = front_nose_count > 0
        engine_regions_detected = int(left_detected) + int(right_detected)

        if engine_regions_detected < min_eng:
            hypothesis_result = "FAIL"
            hypothesis_reason = f"engine_regions_detected<{min_eng}"
        elif require_front and not front_detected:
            hypothesis_result = "FAIL"
            hypothesis_reason = "front_gear_not_detected"
        else:
            hypothesis_result = "PASS"
            hypothesis_reason = "ok"

        print(
            f"[region-det] {scene}: "
            f"left_clusters={left_count} right_clusters={right_count} "
            f"front_nose_clusters={front_nose_count} front_main_clusters={front_main_count} "
            f"-> {hypothesis_result}"
        )
        print(
            f"  centers left={_fmt_centers(left_centers)} "
            f"right={_fmt_centers(right_centers)} "
            f"front_nose={_fmt_centers(front_nose_centers)} "
            f"front_main={_fmt_centers(front_main_centers)}"
        )
        if not front_pcd.exists():
            print("  note: front_gear region PCD missing, so gear clusters are unavailable.")

        rows.append(
            {
                "unique_scene": scene,
                "left_pcd_exists": int(left_pcd.exists()),
                "right_pcd_exists": int(right_pcd.exists()),
                "front_pcd_exists": int(front_pcd.exists()),
                "left_candidates": left_count,
                "right_candidates": right_count,
                "front_nose_candidates": front_nose_count,
                "front_main_candidates": front_main_count,
                "left_detected": int(left_detected),
                "right_detected": int(right_detected),
                "front_detected": int(front_detected),
                "engine_regions_detected": engine_regions_detected,
                "min_engines_required": min_eng,
                "require_front_gear": int(require_front),
                "hypothesis_result": hypothesis_result,
                "hypothesis_reason": hypothesis_reason,
                "left_error": left_error,
                "right_error": right_error,
                "front_error": front_error,
            }
        )
        vis_rows.append(
            {
                "scene": scene,
                "left_pcd": left_pcd,
                "right_pcd": right_pcd,
                "front_pcd": front_pcd,
                "left_clusters": left_clusters,
                "right_clusters": right_clusters,
                "front_nose_clusters": front_nose_clusters,
                "front_main_clusters": front_main_clusters,
                "left_centers": left_centers,
                "right_centers": right_centers,
                "front_nose_centers": front_nose_centers,
                "front_main_centers": front_main_centers,
            }
        )

    out_csv = out_root / "region_hypothesis.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "unique_scene",
        "left_pcd_exists",
        "right_pcd_exists",
        "front_pcd_exists",
        "left_candidates",
        "right_candidates",
        "front_nose_candidates",
        "front_main_candidates",
        "left_detected",
        "right_detected",
        "front_detected",
        "engine_regions_detected",
        "min_engines_required",
        "require_front_gear",
        "hypothesis_result",
        "hypothesis_reason",
        "left_error",
        "right_error",
        "front_error",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for r in rows:
            w.writerow([r.get(k, "") for k in headers])

    if bool(visualize_outputs):
        _visualize_region_hypothesis_rows(
            vis_rows,
            max_scenes=int(vis_max_scenes),
            center_sphere_radius=float(center_sphere_radius),
            split_windows=bool(split_windows),
        )

    centers_csv = out_root / "labeled_region_centers.csv"
    _write_labeled_region_centers_csv(
        vis_rows=vis_rows,
        out_csv=centers_csv,
    )
    print(f"[summary] labeled region centers CSV: {centers_csv}")

    pass_count = sum(
        1 for r in rows
        if str(r.get("hypothesis_result", "")).upper() == "PASS"
    )
    return out_csv, len(rows), int(pass_count)


def _run_region_hypothesis_via_python(
    *,
    python_exe: Path,
    unique_scenes: List[str],
    engine_region_root: Path,
    out_root: Path,
    aircraft_pipeline_root: Path,
    min_engines: int,
    require_front_gear: bool,
    visualize_outputs: bool,
    vis_max_scenes: int,
    center_sphere_radius: float,
    split_windows: bool,
) -> Tuple[Path, int, int]:
    py = Path(python_exe).expanduser().resolve()
    if not py.exists() or not py.is_file():
        raise FileNotFoundError(f"region_hypothesis_python not found: {py}")

    out_csv = out_root / "region_hypothesis.csv"
    unique = sorted(set(unique_scenes))

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", prefix="region_hyp_scenes_", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write("\n".join(unique))
        scenes_path = Path(tmp.name).resolve()

    cmd = [
        str(py),
        "-c",
        _REGION_HYPOTHESIS_SUBPROCESS_CODE,
        "--pipeline-root",
        str(Path(aircraft_pipeline_root).expanduser().resolve()),
        "--engine-region-root",
        str(Path(engine_region_root).expanduser().resolve()),
        "--out-csv",
        str(out_csv),
        "--scenes-file",
        str(scenes_path),
        "--min-engines",
        str(int(max(1, int(min_engines)))),
        "--require-front-gear",
        ("1" if bool(require_front_gear) else "0"),
        "--visualize",
        ("1" if bool(visualize_outputs) else "0"),
        "--vis-max-scenes",
        str(int(max(0, int(vis_max_scenes)))),
        "--center-sphere-radius",
        str(float(max(0.01, float(center_sphere_radius)))),
        "--split-windows",
        ("1" if bool(split_windows) else "0"),
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    finally:
        try:
            scenes_path.unlink(missing_ok=True)
        except Exception:
            pass

    if str(proc.stdout or "").strip():
        print("[region-det-subprocess] stdout:")
        for ln in str(proc.stdout).splitlines():
            if str(ln).strip():
                print(f"  {ln}")
    if str(proc.stderr or "").strip():
        print("[region-det-subprocess] stderr:")
        for ln in str(proc.stderr).splitlines():
            if str(ln).strip():
                print(f"  {ln}")

    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "").strip()
        if "No module named 'open3d'" in msg:
            raise RuntimeError(
                "region hypothesis subprocess is missing open3d. "
                "Install open3d in the --region-hypothesis-python environment."
            )
        raise RuntimeError(
            f"region hypothesis subprocess failed (exit={proc.returncode}): {msg}"
        )

    if not out_csv.exists():
        raise RuntimeError(f"region hypothesis CSV was not created: {out_csv}")

    total = 0
    passed = 0
    with out_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            total += 1
            if str(row.get("hypothesis_result", "")).strip().upper() == "PASS":
                passed += 1
    return out_csv, int(total), int(passed)


def _read_pcd_xyz_ascii(path: Path) -> np.ndarray:
    p = Path(path).expanduser().resolve()
    if not p.exists() or not p.is_file():
        return np.empty((0, 3), dtype=np.float32)
    lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    data_idx = -1
    for i, ln in enumerate(lines):
        s = str(ln).strip().lower()
        if s.startswith("data") and "ascii" in s:
            data_idx = i + 1
            break
    if data_idx < 0 or data_idx >= len(lines):
        return np.empty((0, 3), dtype=np.float32)

    pts: List[List[float]] = []
    for ln in lines[data_idx:]:
        s = str(ln).strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) < 3:
            continue
        try:
            x = float(parts[0])
            y = float(parts[1])
            z = float(parts[2])
        except Exception:
            continue
        if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
            pts.append([x, y, z])
    if not pts:
        return np.empty((0, 3), dtype=np.float32)
    return np.asarray(pts, dtype=np.float32).reshape(-1, 3)


def _write_pcd_xyz_ascii(path: Path, points_xyz: np.ndarray) -> None:
    pts = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    path.parent.mkdir(parents=True, exist_ok=True)
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
        if pts.shape[0] > 0:
            np.savetxt(f, pts, fmt="%.6f %.6f %.6f")


def _build_merger_region_only_scene_pcd(
    *,
    scene_name: str,
    engine_region_root: Path,
    dst_pcd: Path,
) -> int:
    region_root = Path(engine_region_root).expanduser().resolve()

    engine_paths = [
        region_root / "engine_left" / f"{scene_name}.pcd",
        region_root / "engine_right" / f"{scene_name}.pcd",
    ]

    chunks = []

    for fp in engine_paths:
        pts = _read_pcd_xyz_ascii(fp)
        print("engine input:", fp, "points=", pts.shape[0])
        if pts.shape[0] > 0:
            chunks.append(pts)

    if not chunks:
        return 0

    merged = np.concatenate(chunks, axis=0).astype(np.float32)
    merged = merged[np.all(np.isfinite(merged), axis=1)]

    if merged.shape[0] <= 0:
        return 0

    _write_pcd_xyz_ascii(dst_pcd, merged)
    return int(merged.shape[0])


def _build_merger_pose_input_dir(
    *,
    scene_pcd_paths: List[Path],
    out_root: Path,
    use_region_pcd_only: bool,
    engine_region_root: Optional[Path],
) -> Path:
    in_root = out_root / "merger_pose_input"
    if in_root.exists():
        shutil.rmtree(in_root, ignore_errors=True)
    in_root.mkdir(parents=True, exist_ok=True)

    region_root: Optional[Path] = None
    if bool(use_region_pcd_only):
        if engine_region_root is None:
            raise RuntimeError(
                "merger_pose_use_region_pcd_only=on but engine_region_root is unavailable"
            )
        region_root = Path(engine_region_root).expanduser().resolve()

    used_region = 0
    used_full_fallback = 0
    skipped = 0

    for p in scene_pcd_paths:
        src = Path(p).expanduser().resolve()
        scene_name = str(src.stem)
        print("MERGER INPUT PARTS")
        print("left:", engine_region_root / "engine_left" / f"{scene_name}.pcd")
        print("right:", engine_region_root / "engine_right" / f"{scene_name}.pcd")
        print("front:", engine_region_root / "front_gear" / f"{scene_name}.pcd")
        dst = in_root / src.name

        if bool(use_region_pcd_only) and region_root is not None:
            n_region = 0
            try:
                n_region = _build_merger_region_only_scene_pcd(
                    scene_name=scene_name,
                    engine_region_root=region_root,
                    dst_pcd=dst,
                )
            except Exception as e:
                print(
                    f"  [merger-input-warn] scene={scene_name} region-only build failed: "
                    f"{type(e).__name__}: {e}"
                )
                n_region = 0
            if n_region > 0:
                # Also copy front gear separately for nose detection
                front_src = region_root / "front_gear" / f"{scene_name}.pcd"
                front_dst_dir = in_root / "front_gear"
                front_dst_dir.mkdir(parents=True, exist_ok=True)

                if front_src.exists():
                    front_dst = front_dst_dir / f"{scene_name}.pcd"
                    shutil.copy2(front_src, front_dst)
                    print("front gear input:", front_src, "->", front_dst)
                else:
                    print("front gear input missing:", front_src)

                used_region += 1
                continue

        if not src.exists() or not src.is_file():
            skipped += 1
            continue
        try:
            dst.symlink_to(src)
        except Exception:
            shutil.copy2(src, dst)
        used_full_fallback += 1

    if bool(use_region_pcd_only):
        print(
            "[pipeline] merger input build: "
            f"region_only={bool(use_region_pcd_only)} "
            f"region_scenes={int(used_region)} "
            f"full_fallback_scenes={int(used_full_fallback)} "
            f"skipped={int(skipped)}"
        )
    return in_root


def _copy_merger_pose_csv_to_fixed_path(
    *,
    src_csv: Path,
    dst_csv: Path,
) -> None:
    src = Path(src_csv).expanduser().resolve()
    dst = Path(dst_csv).expanduser().resolve()
    if not src.exists() or not src.is_file():
        raise FileNotFoundError(f"source merger CSV not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _run_merger_pose_via_python(
    *,
    python_exe: Path,
    aircraft_pipeline_root: Path,
    pcd_input_root: Path,
    out_root: Path,
    detect_all_parts: bool,
    require_final_graph: bool,
    visualize: bool,
    show_detected_parts: bool,
    log_mode: str,
) -> Tuple[Path, int, int]:
    mode = _normalize_log_mode(str(log_mode), MERGER_POSE_LOG_MODE)
    py = Path(python_exe).expanduser().resolve()
    if not py.exists() or not py.is_file():
        raise FileNotFoundError(f"merger_pose_python not found: {py}")
    if not _python_can_import_module(py, "open3d"):
        try:
            retry_py = _resolve_merger_pose_python(
                explicit_python=None,
                region_python=None,
            )
            if str(retry_py) != str(py):
                if mode == "full":
                    print(
                        f"[merger-subprocess] switching python (open3d missing): "
                        f"{py} -> {retry_py}"
                    )
                py = Path(retry_py).expanduser().resolve()
        except Exception:
            pass

    out_csv = out_root / "merger_pose_hypothesis.csv"
    def _run_with(py_exec: Path) -> subprocess.CompletedProcess:
        cmd = [
            str(py_exec),
            "-c",
            _MERGER_POSE_SUBPROCESS_CODE,
            "--pipeline-root",
            str(Path(aircraft_pipeline_root).expanduser().resolve()),
            "--pcd-root",
            str(Path(pcd_input_root).expanduser().resolve()),
            "--out-csv",
            str(out_csv),
            "--detect-all-parts",
            ("1" if bool(detect_all_parts) else "0"),
            "--require-final-graph",
            ("1" if bool(require_final_graph) else "0"),
            "--visualize",
            ("1" if bool(visualize) else "0"),
            "--show-detected-parts",
            ("1" if bool(show_detected_parts) else "0"),
            "--log-mode",
            str(mode),
        ]
        if mode == "full":
            print(f"[merger-subprocess] python={py_exec}")
        proc_i = subprocess.run(cmd, capture_output=True, text=True)
        if str(proc_i.stdout or "").strip():
            if mode == "full":
                print("[merger-subprocess] stdout:")
                for ln in str(proc_i.stdout).splitlines():
                    if str(ln).strip():
                        print(f"  {ln}")
            elif mode == "minimal":
                for ln in str(proc_i.stdout).splitlines():
                    s = str(ln).strip()
                    if s.startswith("[MERGER-POSE]"):
                        _emit_console(s)
        if mode == "full" and str(proc_i.stderr or "").strip():
            print("[merger-subprocess] stderr:")
            for ln in str(proc_i.stderr).splitlines():
                s = str(ln).strip()
                if s:
                    print(f"  {s}")
        return proc_i

    proc = _run_with(py)
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "").strip()
        if "No module named 'open3d'" in msg:
            retry_py: Optional[Path] = None
            try:
                retry_py = _resolve_merger_pose_python(
                    explicit_python=None,
                    region_python=None,
                )
            except Exception:
                retry_py = None
            if retry_py is not None and str(retry_py) != str(py):
                if mode == "full":
                    print(f"[merger-subprocess] retrying with Open3D python: {retry_py}")
                proc = _run_with(retry_py)
                py = retry_py
                msg = (proc.stderr or proc.stdout or "").strip()
        if proc.returncode != 0:
            if "No module named 'open3d'" in msg:
                if mode == "full":
                    raise RuntimeError(
                        "merger pose subprocess is missing open3d. "
                        f"python={py}. "
                        "Install open3d in the --merger-pose-python environment."
                    )
                raise RuntimeError(
                    "merger pose subprocess failed: open3d missing "
                    f"(python={py})."
                )
            if mode == "full":
                raise RuntimeError(
                    f"merger pose subprocess failed (exit={proc.returncode}) "
                    f"python={py}: {msg}"
                )
            raise RuntimeError(
                f"merger pose subprocess failed (exit={proc.returncode}) python={py}"
            )

    if not out_csv.exists():
        raise RuntimeError(f"merger pose CSV was not created: {out_csv}")

    total = 0
    passed = 0
    with out_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            total += 1
            if str(row.get("pose_result", "")).strip().upper() == "PASS":
                passed += 1
    if total <= 0:
        if mode == "full":
            print("[merger-subprocess] no scene records returned; pose window may not appear.")
    return out_csv, int(total), int(passed)


def main() -> None:
    args = parse_args()
    quiet = _to_bool_switch(str(getattr(args, "quiet", "off")), SUPPRESS_PRINTS)
    _configure_print_suppression(quiet)

    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError("ultralytics is required. Install with `pip install ultralytics`.") from e

    try:
        import test_yolo_pose_from_h5_weights_to_pcd as pose_pcd
        import view_pcd_dir as pcd_view
    except Exception as e:
        raise RuntimeError(
            "Failed to import project modules. Run with project venv activated."
        ) from e

    image_path_raw = str(args.image_path or "").strip()
    image_dir_raw = str(args.image_dir or "").strip()
    weights_raw = str(args.weights or "").strip()
    source_root = str(args.source or "").strip()
    out_dir = str(args.out or "").strip()
    if not weights_raw:
        raise RuntimeError("Please provide --weights.")
    if not source_root:
        raise RuntimeError("Please provide --source.")
    if not out_dir:
        raise RuntimeError("Please provide --out.")
    if image_path_raw and image_dir_raw:
        raise RuntimeError("Use only one of --image-path or --image-dir.")

    visualize = _to_bool_switch(args.visualize, VISUALIZE_PCD)
    visualize_failed_only = _to_bool_switch(
        args.visualize_failed_only, VISUALIZE_FAILED_SCENES_ONLY
    )
    warning_check = _to_bool_switch(args.warning_check, RUN_WARNING_CHECK)
    warning_pass_fail = _to_bool_switch(args.warning_pass_fail, RUN_WARNING_PASS_FAIL)
    use_scene_h5_transform = _to_bool_switch(args.use_scene_h5_transform, USE_SCENE_H5_TRANSFORM)
    save_debug_image = _to_bool_switch(args.save_debug_image, SAVE_DEBUG_IMAGE)
    save_engine_region_pcd = _to_bool_switch(
        args.save_engine_region_pcd, SAVE_ENGINE_REGION_PCD
    )
    show_engine_regions_3d = _to_bool_switch(
        args.show_engine_regions_3d, SHOW_ENGINE_REGION_OVERLAY_3D
    )
    use_engine_region_ratio_for_passfail = _to_bool_switch(
        args.use_engine_region_ratio, USE_ENGINE_REGION_RATIO_FOR_PASSFAIL
    )
    engine_region_inside_ratio_thr = float(
        np.clip(float(args.engine_region_inside_ratio_thr), 0.0, 1.0)
    )
    draw_proxy_keypoints = _to_bool_switch(
        args.draw_proxy_keypoints, DRAW_PROXY_KEYPOINTS_ON_DEBUG_IMAGE
    )
    check_bbox_coverage = _to_bool_switch(args.check_bbox_coverage, CHECK_BBOX_COVERAGE)
    include_front = _to_bool_switch(args.include_front_proxy, INCLUDE_FRONT_PROXY_KEYPOINT)
    use_front_bbox_proxy = _to_bool_switch(args.use_front_bbox_proxy, USE_FRONT_BBOX_FOR_FRONT_PROXY)
    front_class_id = int(args.front_class_id)
    force_engine_side_by_image_x = _to_bool_switch(
        args.force_engine_side_by_image_x, FORCE_ENGINE_SIDE_BY_IMAGE_X
    )
    front_proxy_name = str(args.front_proxy_name or "").strip() or str(FRONT_PROXY_KEYPOINT_NAME)
    failed_scenes_mode = str(args.failed_scenes_mode).strip().lower()
    open_failed_limit = max(1, int(args.open_failed_limit))
    run_region_hypothesis = _to_bool_switch(
        args.run_region_hypothesis, RUN_REGION_HYPOTHESIS
    )
    aircraft_pipeline_root = Path(
        str(args.aircraft_pipeline_root or DEFAULT_AIRCRAFT_PIPELINE_ROOT)
    ).expanduser()
    region_hypothesis_min_engines = max(1, int(args.region_hypothesis_min_engines))
    region_hypothesis_require_front_gear = _to_bool_switch(
        args.region_hypothesis_require_front_gear, REGION_HYPOTHESIS_REQUIRE_FRONT_GEAR
    )
    region_hypothesis_python_raw = str(args.region_hypothesis_python or "").strip()
    region_hypothesis_python = (
        Path(region_hypothesis_python_raw).expanduser()
        if region_hypothesis_python_raw
        else None
    )
    region_hypothesis_visualize_outputs = bool(VISUALIZE_REGION_HYPOTHESIS_OUTPUTS)
    region_hypothesis_split_windows = bool(VISUALIZE_REGION_HYPOTHESIS_SPLIT_WINDOWS)
    region_hypothesis_vis_max_scenes = int(REGION_HYPOTHESIS_VIS_MAX_SCENES)
    region_hypothesis_center_sphere_radius = float(
        REGION_HYPOTHESIS_CENTER_SPHERE_RADIUS
    )
    run_merger_pose = _to_bool_switch(args.run_merger_pose, RUN_MERGER_POSE)
    merger_pose_detect_all_parts = _to_bool_switch(
        args.merger_pose_detect_all_parts, MERGER_POSE_DETECT_ALL_PARTS
    )
    merger_pose_require_final_graph = _to_bool_switch(
        args.merger_pose_require_final_graph, MERGER_POSE_REQUIRE_FINAL_GRAPH
    )
    merger_pose_use_region_pcd_only = _to_bool_switch(
        args.merger_pose_use_region_pcd_only, MERGER_POSE_USE_REGION_PCD_ONLY
    )
    merger_pose_python_raw = str(args.merger_pose_python or "").strip()
    merger_pose_python = (
        Path(merger_pose_python_raw).expanduser()
        if merger_pose_python_raw
        else None
    )
    merger_pose_visualize = _to_bool_switch(
        args.merger_pose_visualize, MERGER_POSE_VISUALIZE
    )
    merger_pose_show_detected_parts = _to_bool_switch(
        args.merger_pose_show_detected_parts, MERGER_POSE_SHOW_DETECTED_PARTS
    )
    merger_pose_log_mode = _normalize_log_mode(
        str(getattr(args, "merger_pose_log_mode", MERGER_POSE_LOG_MODE)),
        MERGER_POSE_LOG_MODE,
    )
    save_merger_pose_fixed_csv = bool(SAVE_MERGER_POSE_FIXED_CSV)
    merger_pose_fixed_csv_path = Path(
        str(MERGER_POSE_FIXED_CSV_PATH or "").strip()
    ).expanduser()
    run_ransac_ground_removal = _to_bool_switch(
        args.ransac_ground_removal, RUN_RANSAC_GROUND_REMOVAL
    )
    ransac_ground_dist_thr = float(max(1e-6, float(args.ransac_ground_dist_thr)))
    ransac_ground_iters = int(max(10, int(args.ransac_ground_iters)))
    ransac_ground_min_inlier_ratio = float(
        np.clip(float(args.ransac_ground_min_inlier_ratio), 0.0, 1.0)
    )
    ransac_ground_min_abs_z = float(np.clip(float(args.ransac_ground_min_abs_z), 0.0, 1.0))
    bbox_margin_m = float(max(0.0, float(args.bbox_margin_m)))
    part_bbox_margin_m = float(max(0.0, float(args.part_bbox_margin_m)))
    part_region_z_expand_m = float(max(0.0, float(args.part_region_z_expand_m)))
    ransac_keep_original_if_empty = bool(RANSAC_KEEP_ORIGINAL_IF_EMPTY)
    strict_per_scene_chaining = bool(STRICT_PER_SCENE_CHAINING)
    chain_pcd_visualize = bool(visualize and VIS_CHAIN_PCD_VIEW)
    chain_engine_region_detector_visualize = bool(
        visualize and VIS_CHAIN_ENGINE_REGION_DETECTOR
    )
    chain_region_hypothesis_visualize_outputs = bool(
        region_hypothesis_visualize_outputs and VIS_CHAIN_REGION_HYPOTHESIS
    )
    chain_merger_pose_visualize = bool(merger_pose_visualize and VIS_CHAIN_MERGER_POSE)
    batch_visualize = bool(visualize and VIS_BATCH_PCD_VIEW and (not strict_per_scene_chaining))
    batch_region_hypothesis_visualize_outputs = bool(
        region_hypothesis_visualize_outputs
        and VIS_BATCH_REGION_HYPOTHESIS
        and (not strict_per_scene_chaining)
    )
    batch_merger_pose_visualize = bool(
        merger_pose_visualize and VIS_BATCH_MERGER_POSE and (not strict_per_scene_chaining)
    )
    if run_merger_pose:
        merger_pose_python = _resolve_merger_pose_python(
            explicit_python=merger_pose_python,
            region_python=region_hypothesis_python,
        )
        print(f"[pipeline] merger_pose_python_resolved={merger_pose_python}")

    out_root = Path(out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    dbg_root = out_root / "debug_imgs"
    if save_debug_image:
        dbg_root.mkdir(parents=True, exist_ok=True)
    if use_engine_region_ratio_for_passfail and not save_engine_region_pcd:
        print(
            "  [warn] save_engine_region_pcd=off but "
            "use_engine_region_ratio=on; forcing save_engine_region_pcd=on"
        )
        save_engine_region_pcd = True
    if run_region_hypothesis and not save_engine_region_pcd:
        print(
            "  [warn] save_engine_region_pcd=off but "
            "run_region_hypothesis=on; forcing save_engine_region_pcd=on"
        )
        save_engine_region_pcd = True
    if run_merger_pose and merger_pose_use_region_pcd_only and not save_engine_region_pcd:
        print(
            "  [warn] save_engine_region_pcd=off but "
            "merger_pose_use_region_pcd_only=on; forcing save_engine_region_pcd=on"
        )
        save_engine_region_pcd = True
    engine_region_root = out_root / "engine_regions"
    if save_engine_region_pcd:
        (engine_region_root / "engine_left").mkdir(parents=True, exist_ok=True)
        (engine_region_root / "engine_right").mkdir(parents=True, exist_ok=True)
        (engine_region_root / "front_gear").mkdir(parents=True, exist_ok=True)

    slot_names: List[str] = ["engine_left_box_center", "engine_right_box_center"]
    if include_front:
        slot_names = [str(front_proxy_name)] + slot_names

    image_paths: List[Path] = []
    if image_dir_raw:
        image_dir = Path(image_dir_raw).expanduser().resolve()
        image_paths = [p.resolve() for p in pose_pcd._collect_images_from_dir(image_dir)]
    elif image_path_raw:
        p = Path(image_path_raw).expanduser().resolve()
        if p.exists() and p.is_dir():
            image_paths = [q.resolve() for q in pose_pcd._collect_images_from_dir(p)]
        else:
            image_paths = [pose_pcd._resolve_image_path_with_split_fallback(image_path_raw).resolve()]
    else:
        raise RuntimeError("Please provide --image-path or --image-dir.")

    max_images = int(args.max_images)
    if max_images > 0:
        image_paths = image_paths[:max_images]
    if not image_paths:
        raise RuntimeError("No images selected for processing.")

    print(f"[pipeline] images requested: {len(image_paths)}")
    print(f"[pipeline] source H5 root: {source_root}")
    print(f"[pipeline] output: {out_root}")
    print(
        "[pipeline] ransac-ground-removal: "
        f"enabled={bool(run_ransac_ground_removal)} "
        f"dist_thr={float(ransac_ground_dist_thr):.3f} "
        f"iters={int(ransac_ground_iters)} "
        f"min_inlier_ratio={float(ransac_ground_min_inlier_ratio):.3f} "
        f"min_abs_z={float(ransac_ground_min_abs_z):.3f}"
    )
    print(
        f"[pipeline] bbox metric margin (Euclidean radius): "
        f"{float(bbox_margin_m):.3f} m"
    )
    print(
        f"[pipeline] part-region bbox metric margin (Euclidean radius): "
        f"{float(part_bbox_margin_m):.3f} m"
    )
    print(
        f"[pipeline] part-region vertical expand (+/-z): "
        f"{float(part_region_z_expand_m):.3f} m"
    )
    print(f"[pipeline] force_engine_side_by_image_x={bool(force_engine_side_by_image_x)}")
    print(
        f"[pipeline] merger_pose_use_region_pcd_only="
        f"{bool(merger_pose_use_region_pcd_only)}"
    )

    print("[list] Searching for .h5 files...")
    h5_paths = list_h5_paths(source_root)
    if not h5_paths:
        raise RuntimeError(f"No .h5 files found under: {source_root}")

    by_h5_stem: Dict[str, List[str]] = defaultdict(list)
    for hp in h5_paths:
        by_h5_stem[Path(hp).stem].append(hp)

    scene_requests_by_h5: Dict[str, List[SceneRequest]] = defaultdict(list)
    unique_scenes_requested: List[str] = []
    unique_scene_to_source_image: Dict[str, Path] = {}
    skipped_bad_stem = 0
    skipped_unreadable = 0
    skipped_missing_h5 = 0

    for ip in image_paths:
        bgr = cv2.imread(str(ip))
        if bgr is None:
            skipped_unreadable += 1
            print(f"[skip] unreadable image: {ip}")
            continue
        try:
            h5_stem, scene_name = pose_pcd._parse_unique_scene_stem(ip.stem)
        except Exception as e:
            skipped_bad_stem += 1
            print(f"[skip] bad image stem: {ip.name} ({e})")
            continue

        matches = by_h5_stem.get(h5_stem, [])
        if not matches:
            skipped_missing_h5 += 1
            print(f"[skip] no H5 match for stem='{h5_stem}' image={ip.name}")
            continue
        if len(matches) > 1:
            print(f"[warn] multiple H5 matches for '{h5_stem}', using first: {matches[0]}")

        h5_match = matches[0]
        scene_requests_by_h5[h5_match].append(
            SceneRequest(
                scene_name=str(scene_name),
                image_path=ip,
                image_bgr=bgr,
            )
        )
        unique_scene = f"{h5_stem}__{scene_name}"
        unique_scenes_requested.append(unique_scene)
        unique_scene_to_source_image[unique_scene] = ip

    if not scene_requests_by_h5:
        raise RuntimeError("No valid image->H5 scene mappings found.")

    print(
        "[mapping] "
        f"mapped={len(unique_scenes_requested)} "
        f"skipped_unreadable={skipped_unreadable} "
        f"skipped_bad_stem={skipped_bad_stem} "
        f"skipped_missing_h5={skipped_missing_h5}"
    )

    weights_path = pose_pcd._resolve_weights(weights_raw)
    print(f"[model] using weights: {weights_path}")
    model = YOLO(str(weights_path))

    kp_conf_rows: List[Dict[str, Any]] = []
    bbox_cov_rows: List[Dict[str, Any]] = []
    total_saved_pcd = 0
    total_saved_dbg = 0
    skipped_no_bbox = 0
    skipped_invalid_bbox = 0
    skipped_empty_pcd = 0
    skipped_scene_missing = 0
    skipped_missing_xyz = 0
    total_engine_left_region_saved = 0
    total_engine_right_region_saved = 0
    total_front_region_saved = 0
    total_ground_removed_main = 0
    total_ground_removed_regions = 0

    for h5_idx, (h5_path, reqs) in enumerate(scene_requests_by_h5.items(), 1):
        print(f"[{h5_idx}/{len(scene_requests_by_h5)}] {Path(h5_path).name} scenes={len(reqs)}")
        try:
            with open_h5_any(h5_path) as f:
                H = int(f.attrs["height"])
                W = int(f.attrs["width"])

                for req in reqs:
                    scene_name = str(req.scene_name)
                    unique_scene = f"{Path(h5_path).stem}__{scene_name}"
                    if scene_name not in f:
                        skipped_scene_missing += 1
                        print(f"  [skip] scene not in H5: {unique_scene}")
                        continue
                    grp = f[scene_name]
                    if not isinstance(grp, h5py.Group) or "points" not in grp:
                        skipped_scene_missing += 1
                        print(f"  [skip] scene has no points group: {unique_scene}")
                        continue

                    flat, cols = _load_scene_points_and_meta(grp)
                    rgb_h5, xyz_hw3 = pose_pcd._build_rgb_and_xyz(flat, cols, H, W)
                    if rgb_h5 is None or xyz_hw3 is None:
                        skipped_missing_xyz += 1
                        print(f"  [skip] cannot build xyz/rgb: {unique_scene}")
                        if check_bbox_coverage:
                            bbox_cov_rows.append(
                                {
                                    "unique_scene": unique_scene,
                                    "h5_file": Path(h5_path).name,
                                    "scene_name": scene_name,
                                    "bbox_x1": None,
                                    "bbox_y1": None,
                                    "bbox_x2": None,
                                    "bbox_y2": None,
                                    "bbox_status": "FAIL",
                                    "bbox_reason": "missing_required_cols",
                                    "aircraft_px_total": None,
                                    "aircraft_px_inside": None,
                                    "aircraft_recall": None,
                                    "bbox_area_px": None,
                                    "full_threshold": float(args.bbox_full_thr),
                                }
                            )
                        continue

                    mask_aircraft_eval = pose_pcd._extract_is_aircraft_mask(flat, cols, H, W)
                    if mask_aircraft_eval is not None:
                        shift_cols = pose_pcd._compute_export_like_roll(mask_aircraft_eval)
                        if shift_cols != 0:
                            xyz_hw3 = np.roll(xyz_hw3, shift=shift_cols, axis=1)
                            rgb_h5 = np.roll(rgb_h5, shift=shift_cols, axis=1)
                            mask_aircraft_eval = np.roll(mask_aircraft_eval, shift=shift_cols, axis=1)
                    scene_pts = np.asarray(xyz_hw3, dtype=np.float32).reshape(-1, 3)
                    finite_scene = np.all(np.isfinite(scene_pts), axis=1)
                    scene_finite_xyz = scene_pts[finite_scene]

                    rgb_for_model = req.image_bgr
                    if rgb_for_model.shape[0] != H or rgb_for_model.shape[1] != W:
                        print(
                            f"  [image] resize {rgb_for_model.shape[1]}x{rgb_for_model.shape[0]} -> {W}x{H} ({unique_scene})"
                        )
                        rgb_for_model = cv2.resize(rgb_for_model, (W, H), interpolation=cv2.INTER_LINEAR)

                    results_img = model.predict(
                        rgb_for_model,
                        imgsz=int(args.imgsz),
                        conf=float(args.conf),
                        device=str(args.device),
                        verbose=False,
                    )
                    det_img: Optional[Dict[str, Any]] = None
                    if results_img:
                        det_img = _extract_scene_detections_from_result(
                            results_img[0],
                            image_w=int(W),
                            image_h=int(H),
                            front_class_id=int(front_class_id),
                            force_engine_side_by_image_x=bool(force_engine_side_by_image_x),
                        )

                    # Optional fallback: run detector on rolled H5 RGB (guaranteed XYZ-aligned),
                    # then choose the better candidate by aircraft-mask coverage + engine count.
                    use_h5_fallback = bool(TRY_H5_RGB_DETECTION_FALLBACK)
                    det_h5: Optional[Dict[str, Any]] = None
                    det_selected: Optional[Dict[str, Any]] = det_img
                    det_source = "image"
                    if use_h5_fallback:
                        img_engine_count = 0
                        img_recall = -1.0
                        if det_img is not None:
                            img_engine_count = int(det_img["eng_left_bb"] is not None) + int(
                                det_img["eng_right_bb"] is not None
                            )
                            if mask_aircraft_eval is not None:
                                cov_img = pose_pcd._bbox_aircraft_coverage(
                                    mask_aircraft_eval,
                                    det_img["aircraft_bb"],
                                )
                                if cov_img is not None:
                                    img_recall = float(cov_img["aircraft_recall"])

                        need_retry = bool(det_img is None)
                        if (
                            not need_retry
                            and bool(H5_RGB_FALLBACK_IF_NO_ENGINE_BBOX)
                            and img_engine_count <= 0
                        ):
                            need_retry = True
                        if (
                            not need_retry
                            and mask_aircraft_eval is not None
                            and img_recall >= 0.0
                            and img_recall < float(H5_RGB_FALLBACK_MIN_AIRCRAFT_RECALL)
                        ):
                            need_retry = True

                        if need_retry:
                            results_h5 = model.predict(
                                rgb_h5,
                                imgsz=int(args.imgsz),
                                conf=float(args.conf),
                                device=str(args.device),
                                verbose=False,
                            )
                            if results_h5:
                                det_h5 = _extract_scene_detections_from_result(
                                    results_h5[0],
                                    image_w=int(W),
                                    image_h=int(H),
                                    front_class_id=int(front_class_id),
                                    force_engine_side_by_image_x=bool(
                                        force_engine_side_by_image_x
                                    ),
                                )

                            def _cand_score(cand: Optional[Dict[str, Any]]) -> Tuple[int, float, int, float]:
                                if cand is None:
                                    return (0, -1.0, 0, -1.0)
                                ec = int(cand["eng_left_bb"] is not None) + int(
                                    cand["eng_right_bb"] is not None
                                )
                                rec = -1.0
                                if mask_aircraft_eval is not None:
                                    cov = pose_pcd._bbox_aircraft_coverage(
                                        mask_aircraft_eval, cand["aircraft_bb"]
                                    )
                                    if cov is not None:
                                        rec = float(cov["aircraft_recall"])
                                return (1, float(rec), int(ec), float(cand["aircraft_conf"]))

                            sc_img = _cand_score(det_img)
                            sc_h5 = _cand_score(det_h5)
                            if sc_h5 > sc_img:
                                det_selected = det_h5
                                det_source = "h5_rgb"
                                rgb_for_model = rgb_h5
                            if det_source == "h5_rgb" or det_h5 is not None:
                                print(
                                    f"  [det-fallback] {unique_scene}: "
                                    f"img_score={sc_img} h5_score={sc_h5} "
                                    f"chosen={det_source}"
                                )

                    if det_selected is None:
                        skipped_no_bbox += 1
                        print(f"  [skip] no yolo result/aircraft bbox: {unique_scene}")
                        if check_bbox_coverage:
                            bbox_cov_rows.append(
                                {
                                    "unique_scene": unique_scene,
                                    "h5_file": Path(h5_path).name,
                                    "scene_name": scene_name,
                                    "bbox_x1": None,
                                    "bbox_y1": None,
                                    "bbox_x2": None,
                                    "bbox_y2": None,
                                    "bbox_status": "FAIL",
                                    "bbox_reason": "no_bbox",
                                    "aircraft_px_total": None,
                                    "aircraft_px_inside": None,
                                    "aircraft_recall": None,
                                    "bbox_area_px": None,
                                    "full_threshold": float(args.bbox_full_thr),
                                }
                            )
                        continue

                    aircraft_bb = det_selected["aircraft_bb"]
                    aircraft_conf = float(det_selected["aircraft_conf"])
                    eng_left_bb = det_selected["eng_left_bb"]
                    eng_left_conf = float(det_selected["eng_left_conf"])
                    eng_right_bb = det_selected["eng_right_bb"]
                    eng_right_conf = float(det_selected["eng_right_conf"])
                    front_bb = det_selected["front_bb"]
                    front_conf = float(det_selected["front_conf"])
                    side_fix_reason = str(det_selected.get("side_fix_reason", "none"))
                    if side_fix_reason != "none":
                        print(f"  [side-fix] {unique_scene}: {side_fix_reason}")
                    if run_region_hypothesis:
                        engine_bbox_count = int(eng_left_bb is not None) + int(eng_right_bb is not None)
                        req_eng = int(region_hypothesis_min_engines)
                        if engine_bbox_count < req_eng:
                            print(
                                f"  [warn] {unique_scene}: detected engine bboxes={engine_bbox_count} "
                                f"but region_hypothesis_min_engines={req_eng}; "
                                "region hypothesis may FAIL."
                            )

                    x1, y1, x2, y2 = aircraft_bb
                    if check_bbox_coverage:
                        cov = pose_pcd._bbox_aircraft_coverage(mask_aircraft_eval, aircraft_bb)
                        if cov is None:
                            bbox_cov_rows.append(
                                {
                                    "unique_scene": unique_scene,
                                    "h5_file": Path(h5_path).name,
                                    "scene_name": scene_name,
                                    "bbox_x1": x1,
                                    "bbox_y1": y1,
                                    "bbox_x2": x2,
                                    "bbox_y2": y2,
                                    "bbox_status": "FAIL",
                                    "bbox_reason": "missing_aircraft_mask",
                                    "aircraft_px_total": None,
                                    "aircraft_px_inside": None,
                                    "aircraft_recall": None,
                                    "bbox_area_px": int((x2 - x1 + 1) * (y2 - y1 + 1)),
                                    "full_threshold": float(args.bbox_full_thr),
                                }
                            )
                        else:
                            rec = float(cov["aircraft_recall"])
                            inside = int(round(float(cov["aircraft_px_inside"])))
                            total = int(round(float(cov["aircraft_px_total"])))
                            is_pass = rec >= float(args.bbox_full_thr)
                            bbox_cov_rows.append(
                                {
                                    "unique_scene": unique_scene,
                                    "h5_file": Path(h5_path).name,
                                    "scene_name": scene_name,
                                    "bbox_x1": x1,
                                    "bbox_y1": y1,
                                    "bbox_x2": x2,
                                    "bbox_y2": y2,
                                    "bbox_status": ("PASS" if is_pass else "FAIL"),
                                    "bbox_reason": ("full_aircraft_covered" if is_pass else "aircraft_outside_bbox"),
                                    "aircraft_px_total": total,
                                    "aircraft_px_inside": inside,
                                    "aircraft_recall": rec,
                                    "bbox_area_px": int(round(float(cov["bbox_area_px"]))),
                                    "full_threshold": float(args.bbox_full_thr),
                                }
                            )

                    pts_raw = _extract_points_from_bbox(
                        xyz_hw3=xyz_hw3,
                        bb=aircraft_bb,
                        metric_margin_m=float(bbox_margin_m),
                        all_finite_xyz=scene_finite_xyz,
                    )
                    if pts_raw.size == 0:
                        skipped_empty_pcd += 1
                        print(f"  [skip] empty pointcloud in aircraft bbox: {unique_scene}")
                        continue

                    pts = np.asarray(pts_raw, dtype=np.float32).reshape(-1, 3)
                    if run_ransac_ground_removal:
                        pts, gr_stats = _remove_ground_ransac(
                            pts,
                            dist_thr=float(ransac_ground_dist_thr),
                            max_iters=int(ransac_ground_iters),
                            min_inlier_ratio=float(ransac_ground_min_inlier_ratio),
                            min_abs_z=float(ransac_ground_min_abs_z),
                            keep_original_if_empty=bool(ransac_keep_original_if_empty),
                        )
                        total_ground_removed_main += int(gr_stats.get("ground_points", 0))
                        if int(gr_stats.get("ground_points", 0)) > 0:
                            print(
                                f"  [ground-ransac] main {unique_scene}: "
                                f"removed={int(gr_stats.get('ground_points', 0))} "
                                f"kept={int(gr_stats.get('kept_points', 0))} "
                                f"fallback={int(gr_stats.get('used_fallback', 0))}"
                            )

                    kp_points: List[np.ndarray] = []
                    kp_slots_px: List[Tuple[str, Tuple[int, int], float]] = []
                    kp_conf_map: Dict[str, float] = {}

                    def _append_slot(
                        name: str,
                        center_xy: Tuple[int, int],
                        conf_v: float,
                        sampled_xyz: Optional[np.ndarray],
                        *,
                        draw_on_debug: bool,
                    ) -> None:
                        p3 = sampled_xyz
                        if p3 is None:
                            print(
                                f"  [warn] no valid xyz at {name} center: "
                                f"{unique_scene} (skipping keypoint)"
                            )
                            return
                        p3 = np.asarray(p3, dtype=np.float32).reshape(3)
                        kp_points.append(p3)
                        if bool(draw_on_debug):
                            kp_slots_px.append((name, center_xy, float(conf_v)))
                        kp_conf_map[name] = float(conf_v)

                    if include_front:
                        if bool(use_front_bbox_proxy):
                            if front_bb is None:
                                print(
                                    f"  [warn] front bbox missing: {unique_scene} "
                                    "(skipping front proxy)"
                                )
                            else:
                                fcx, fcy = _center_px_from_bbox(front_bb)
                                p_front = pose_pcd._sample_xyz_nearest(
                                    xyz_hw3=xyz_hw3,
                                    r0=int(fcy),
                                    c0=int(fcx),
                                    radius=int(args.engine_center_patch_radius),
                                    allow_global_fallback=False,
                                )
                                _append_slot(
                                    name=str(front_proxy_name),
                                    center_xy=(int(fcx), int(fcy)),
                                    conf_v=float(front_conf),
                                    sampled_xyz=p_front,
                                    draw_on_debug=bool(draw_proxy_keypoints),
                                )
                        elif front_bb is not None:
                            fcx, fcy = _center_px_from_bbox(front_bb)
                            p_front = pose_pcd._sample_xyz_nearest(
                                xyz_hw3=xyz_hw3,
                                r0=int(fcy),
                                c0=int(fcx),
                                radius=int(args.engine_center_patch_radius),
                                allow_global_fallback=False,
                            )
                            _append_slot(
                                name=str(front_proxy_name),
                                center_xy=(int(fcx), int(fcy)),
                                conf_v=float(front_conf),
                                sampled_xyz=p_front,
                                draw_on_debug=bool(draw_proxy_keypoints),
                            )
                        else:
                            print(
                                f"  [warn] front bbox missing: {unique_scene} "
                                "(skipping front proxy)"
                            )

                    if eng_left_bb is not None:
                        lcx, lcy = _center_px_from_bbox(eng_left_bb)
                        p_left = pose_pcd._sample_xyz_nearest(
                            xyz_hw3=xyz_hw3,
                            r0=int(lcy),
                            c0=int(lcx),
                            radius=int(args.engine_center_patch_radius),
                            allow_global_fallback=False,
                        )
                        _append_slot(
                            name="engine_left_box_center",
                            center_xy=(int(lcx), int(lcy)),
                            conf_v=float(eng_left_conf),
                            sampled_xyz=p_left,
                            draw_on_debug=bool(draw_proxy_keypoints),
                        )
                    else:
                        print(f"  [warn] engine_left bbox missing: {unique_scene} (skipping engine_left warning check)")

                    if eng_right_bb is not None:
                        rcx, rcy = _center_px_from_bbox(eng_right_bb)
                        p_right = pose_pcd._sample_xyz_nearest(
                            xyz_hw3=xyz_hw3,
                            r0=int(rcy),
                            c0=int(rcx),
                            radius=int(args.engine_center_patch_radius),
                            allow_global_fallback=False,
                        )
                        _append_slot(
                            name="engine_right_box_center",
                            center_xy=(int(rcx), int(rcy)),
                            conf_v=float(eng_right_conf),
                            sampled_xyz=p_right,
                            draw_on_debug=bool(draw_proxy_keypoints),
                        )
                    else:
                        print(f"  [warn] engine_right bbox missing: {unique_scene} (skipping engine_right warning check)")

                    pts_to_save = pts
                    if kp_points:
                        kp_arr = np.stack(kp_points, axis=0).astype(np.float32)
                        pts_to_save = np.concatenate([pts.astype(np.float32), kp_arr], axis=0)

                    out_pcd = out_root / f"{unique_scene}.pcd"
                    pose_pcd.write_pcd_xyz(out_pcd, pts_to_save)
                    total_saved_pcd += 1

                    if save_engine_region_pcd:
                        # Remove stale part-region files from previous runs for this scene.
                        # Otherwise old right/left/front PCDs can be reused accidentally
                        # when current detections are missing on one side.
                        for part_name in ("engine_left", "engine_right", "front_gear"):
                            stale_fp = engine_region_root / part_name / f"{unique_scene}.pcd"
                            try:
                                if stale_fp.exists():
                                    stale_fp.unlink()
                            except Exception:
                                pass
                        pts_left_region = _extract_points_from_bbox(
                            xyz_hw3=xyz_hw3,
                            bb=eng_left_bb,
                            metric_margin_m=float(part_bbox_margin_m),
                            all_finite_xyz=scene_finite_xyz,
                        )
                        pts_right_region = _extract_points_from_bbox(
                            xyz_hw3=xyz_hw3,
                            bb=eng_right_bb,
                            metric_margin_m=float(part_bbox_margin_m),
                            all_finite_xyz=scene_finite_xyz,
                        )
                        pts_front_region = _extract_points_from_bbox(
                            xyz_hw3=xyz_hw3,
                            bb=front_bb,
                            metric_margin_m=float(FRONT_REGION_BBOX_MARGIN_M),
                            all_finite_xyz=scene_finite_xyz,
                            exclude_bbs=[eng_left_bb, eng_right_bb],
                        )
                        if float(part_region_z_expand_m) > 0.0:
                            # pts_left_region = _expand_region_points_vertical(
                            #     pts_left_region,
                            #     all_finite_xyz=scene_finite_xyz,
                            #     z_expand_m=float(part_region_z_expand_m),
                            # )
                            # pts_right_region = _expand_region_points_vertical(
                            #     pts_right_region,
                            #     all_finite_xyz=scene_finite_xyz,
                            #     z_expand_m=float(part_region_z_expand_m),
                            # )
                            pts_front_region = _expand_region_points_vertical(
                                pts_front_region,
                                all_finite_xyz=scene_finite_xyz,
                                z_expand_m=float(part_region_z_expand_m),
                            )
                        if run_ransac_ground_removal:
                            if pts_left_region.shape[0] > 0:
                                pts_left_region, grl = _remove_ground_ransac(
                                    pts_left_region,
                                    dist_thr=float(ransac_ground_dist_thr),
                                    max_iters=int(ransac_ground_iters),
                                    min_inlier_ratio=float(ransac_ground_min_inlier_ratio),
                                    min_abs_z=float(ransac_ground_min_abs_z),
                                    keep_original_if_empty=bool(ransac_keep_original_if_empty),
                                )
                                total_ground_removed_regions += int(grl.get("ground_points", 0))
                            if pts_right_region.shape[0] > 0:
                                pts_right_region, grr = _remove_ground_ransac(
                                    pts_right_region,
                                    dist_thr=float(ransac_ground_dist_thr),
                                    max_iters=int(ransac_ground_iters),
                                    min_inlier_ratio=float(ransac_ground_min_inlier_ratio),
                                    min_abs_z=float(ransac_ground_min_abs_z),
                                    keep_original_if_empty=bool(ransac_keep_original_if_empty),
                                )
                                total_ground_removed_regions += int(grr.get("ground_points", 0))
                            # Front gear / nose gear region:
                            # Do NOT run RANSAC here because nose gear is close to the ground.
                            # RANSAC can remove the real nose-gear points.
                            if pts_front_region.shape[0] > 0:
                                grf = {
                                    "input_points": int(pts_front_region.shape[0]),
                                    "ground_points": 0,
                                    "kept_points": int(pts_front_region.shape[0]),
                                    "used_fallback": 0,
                                }
                        if pts_left_region.shape[0] > 0:
                            out_left = engine_region_root / "engine_left" / f"{unique_scene}.pcd"
                            pose_pcd.write_pcd_xyz(out_left, pts_left_region)
                            total_engine_left_region_saved += 1
                        if pts_right_region.shape[0] > 0:
                            out_right = engine_region_root / "engine_right" / f"{unique_scene}.pcd"
                            pose_pcd.write_pcd_xyz(out_right, pts_right_region)
                            total_engine_right_region_saved += 1
                        if pts_front_region.shape[0] > 0:
                            out_front = engine_region_root / "front_gear" / f"{unique_scene}.pcd"
                            pose_pcd.write_pcd_xyz(out_front, pts_front_region)
                            total_front_region_saved += 1

                    if save_debug_image:
                        dbg = _draw_overlay(
                            image_bgr=rgb_for_model,
                            aircraft_bb=aircraft_bb,
                            eng_left_bb=eng_left_bb,
                            eng_right_bb=eng_right_bb,
                            front_bb=front_bb,
                            kp_slots=kp_slots_px,
                        )
                        cv2.imwrite(str(dbg_root / f"{unique_scene}.png"), dbg)
                        total_saved_dbg += 1

                    kp_conf_rows.append(
                        {
                            "unique_scene": unique_scene,
                            "h5_file": Path(h5_path).name,
                            "scene_name": scene_name,
                            "conf_map": kp_conf_map,
                        }
                    )

        except Exception as e:
            print(f"[error] {Path(h5_path).name}: {e}")

    keypoint_conf_csv = out_root / "keypoint_confidence.csv"
    bbox_cov_csv = out_root / "bbox_aircraft_coverage.csv"
    kp_passfail_csv = out_root / "keypoint_pass_fail_confidence.csv"

    _write_keypoint_conf_csv(rows=kp_conf_rows, out_csv=keypoint_conf_csv, slot_names=slot_names)
    print(f"[summary] keypoint confidence CSV: {keypoint_conf_csv}")

    if check_bbox_coverage:
        _write_bbox_cov_csv(rows=bbox_cov_rows, out_csv=bbox_cov_csv)
        print(f"[summary] bbox coverage CSV: {bbox_cov_csv}")

    pcd_paths = sorted([out_root / f"{s}.pcd" for s in unique_scenes_requested if (out_root / f"{s}.pcd").exists()])
    if not pcd_paths:
        raise RuntimeError("No PCD files were generated.")

    print(f"[summary] scenes requested={len(unique_scenes_requested)}")
    print(f"[summary] pcd_saved={total_saved_pcd}")
    print(f"[summary] debug_images_saved={total_saved_dbg}")
    if save_engine_region_pcd:
        print(
            f"[summary] engine_region_pcd_saved: "
            f"left={total_engine_left_region_saved} right={total_engine_right_region_saved} "
            f"front={total_front_region_saved}"
        )
    if run_ransac_ground_removal:
        print(
            f"[summary] ransac ground removed: "
            f"main={int(total_ground_removed_main)} "
            f"regions={int(total_ground_removed_regions)}"
        )
    print(
        "[summary] skipped: "
        f"no_bbox={skipped_no_bbox} invalid_bbox={skipped_invalid_bbox} "
        f"empty_pcd={skipped_empty_pcd} missing_scene={skipped_scene_missing} missing_xyz={skipped_missing_xyz}"
    )

    if strict_per_scene_chaining:
        print("[pipeline] Per-scene visualization chain: pcd-view -> engine-detector -> region -> pose")
        merger_scene_python = (
            Path(merger_pose_python).expanduser()
            if merger_pose_python is not None
            else None
        )
        for sidx, scene_pcd in enumerate(pcd_paths, 1):
            scene_name = str(scene_pcd.stem)
            print(f"[chain {sidx}/{len(pcd_paths)}] scene={scene_name}")
            try:
                with tempfile.TemporaryDirectory(prefix="scene_chain_") as td:
                    tmp_root = Path(td).resolve()
                    tmp_kp_passfail_csv = tmp_root / "keypoint_pass_fail_confidence.csv"

                    print("  [chain] 1/4 region-detected PCD view")
                    pcd_view._view_files(
                        paths=[scene_pcd],
                        show_axes=bool(SHOW_AXES),
                        visualize=bool(chain_pcd_visualize),
                        kpt_count=0,
                        kpt_radius=0.25,
                        warning_check_enabled=bool(warning_check),
                        warning_pass_fail_enabled=bool(warning_pass_fail and warning_check),
                        warning_keypoint_csv=keypoint_conf_csv,
                        warning_profile_csv=str(pcd_view.WARNING_PROFILE_CSV),
                        warning_yaml_column=str(pcd_view.WARNING_YAML_COLUMN),
                        warning_yaml_root=str(pcd_view.WARNING_YAML_ROOT),
                        warning_yaml_relpath=str(pcd_view.WARNING_YAML_RELPATH),
                        warning_target_level=int(pcd_view.WARNING_TARGET_LEVEL),
                        warning_box_scale=1.0,
                        warning_fallback_kp_names=[],
                        warning_h5_root=str(source_root),
                        use_scene_h5_transform=bool(use_scene_h5_transform),
                        warning_kp_passfail_csv=tmp_kp_passfail_csv,
                        warning_conf_threshold=0.0,
                        infer_kpt_count_from_csv=True,
                        show_kp_spheres=False,
                        tail_points_label="engine_anchors",
                        engine_region_root=(
                            engine_region_root
                            if (show_engine_regions_3d or use_engine_region_ratio_for_passfail)
                            else None
                        ),
                        show_engine_region_points=bool(show_engine_regions_3d),
                        use_engine_region_ratio_for_passfail=bool(
                            use_engine_region_ratio_for_passfail
                        ),
                        engine_region_inside_ratio_thr=float(engine_region_inside_ratio_thr),
                        visualize_failed_scenes_only=False,
                    )

                    if save_engine_region_pcd and bool(chain_engine_region_detector_visualize):
                        print("  [chain] 2/4 engine-detector on engine regions")
                        _visualize_engine_detector_region_outputs(
                            unique_scenes=[scene_name],
                            engine_region_root=engine_region_root,
                            aircraft_pipeline_root=aircraft_pipeline_root,
                            max_scenes=1,
                            center_sphere_radius=float(
                                region_hypothesis_center_sphere_radius
                            ),
                        )

                    if run_region_hypothesis:
                        print("  [chain] 3/4 region-detector clusters+centers")
                        if region_hypothesis_python is None:
                            _run_region_hypothesis(
                                unique_scenes=[scene_name],
                                engine_region_root=engine_region_root,
                                out_root=tmp_root,
                                aircraft_pipeline_root=aircraft_pipeline_root,
                                min_engines=int(region_hypothesis_min_engines),
                                require_front_gear=bool(
                                    region_hypothesis_require_front_gear
                                ),
                                visualize_outputs=bool(
                                    chain_region_hypothesis_visualize_outputs
                                ),
                                vis_max_scenes=1,
                                center_sphere_radius=float(
                                    region_hypothesis_center_sphere_radius
                                ),
                                split_windows=bool(region_hypothesis_split_windows),
                            )
                        else:
                            _run_region_hypothesis_via_python(
                                python_exe=region_hypothesis_python,
                                unique_scenes=[scene_name],
                                engine_region_root=engine_region_root,
                                out_root=tmp_root,
                                aircraft_pipeline_root=aircraft_pipeline_root,
                                min_engines=int(region_hypothesis_min_engines),
                                require_front_gear=bool(
                                    region_hypothesis_require_front_gear
                                ),
                                visualize_outputs=bool(
                                    chain_region_hypothesis_visualize_outputs
                                ),
                                vis_max_scenes=1,
                                center_sphere_radius=float(
                                    region_hypothesis_center_sphere_radius
                                ),
                                split_windows=bool(region_hypothesis_split_windows),
                            )

                    if (
                        run_merger_pose
                        and merger_scene_python is not None
                    ):
                        print("  [chain] 4/4 merger pose hypothesis")
                        scene_merger_input_root = _build_merger_pose_input_dir(
                            scene_pcd_paths=[scene_pcd],
                            out_root=tmp_root,
                            use_region_pcd_only=bool(merger_pose_use_region_pcd_only),
                            engine_region_root=(
                                engine_region_root
                                if bool(merger_pose_use_region_pcd_only)
                                else None
                            ),
                        )
                        _run_merger_pose_via_python(
                            python_exe=merger_scene_python,
                            aircraft_pipeline_root=aircraft_pipeline_root,
                            pcd_input_root=scene_merger_input_root,
                            out_root=tmp_root,
                            detect_all_parts=bool(merger_pose_detect_all_parts),
                            require_final_graph=bool(merger_pose_require_final_graph),
                            visualize=bool(chain_merger_pose_visualize),
                            show_detected_parts=bool(merger_pose_show_detected_parts),
                            log_mode=str(merger_pose_log_mode),
                        )
            except Exception as e:
                print(f"  [chain-warn] scene={scene_name} failed: {type(e).__name__}: {e}")

    print("[pipeline] Step 2/4: PCD view/check (batch outputs)")
    print(
        f"[pipeline] visualize={bool(batch_visualize)} "
        f"warning_check={bool(warning_check)} warning_pass_fail={bool(warning_pass_fail)}"
    )
    print(f"[pipeline] visualize_failed_only={bool(visualize_failed_only)}")
    print(f"[pipeline] use_scene_h5_transform={bool(use_scene_h5_transform)}")
    print(
        "[pipeline] engine-region ratio mode: "
        f"enabled={bool(use_engine_region_ratio_for_passfail)} "
        f"thr={float(engine_region_inside_ratio_thr):.3f}"
    )

    pcd_view._view_files(
        paths=pcd_paths,
        show_axes=bool(SHOW_AXES),
        visualize=bool(batch_visualize),
        kpt_count=0,
        kpt_radius=0.25,
        warning_check_enabled=bool(warning_check),
        warning_pass_fail_enabled=bool(warning_pass_fail and warning_check),
        warning_keypoint_csv=keypoint_conf_csv,
        warning_profile_csv=str(pcd_view.WARNING_PROFILE_CSV),
        warning_yaml_column=str(pcd_view.WARNING_YAML_COLUMN),
        warning_yaml_root=str(pcd_view.WARNING_YAML_ROOT),
        warning_yaml_relpath=str(pcd_view.WARNING_YAML_RELPATH),
        warning_target_level=int(pcd_view.WARNING_TARGET_LEVEL),
        warning_box_scale=1.0,
        warning_fallback_kp_names=[],
        warning_h5_root=str(source_root),
        use_scene_h5_transform=bool(use_scene_h5_transform),
        warning_kp_passfail_csv=kp_passfail_csv,
        warning_conf_threshold=0.0,
        infer_kpt_count_from_csv=True,
        show_kp_spheres=False,
        tail_points_label="engine_anchors",
        engine_region_root=(
            engine_region_root
            if (show_engine_regions_3d or use_engine_region_ratio_for_passfail)
            else None
        ),
        show_engine_region_points=bool(show_engine_regions_3d),
        use_engine_region_ratio_for_passfail=bool(use_engine_region_ratio_for_passfail),
        engine_region_inside_ratio_thr=float(engine_region_inside_ratio_thr),
        visualize_failed_scenes_only=bool(visualize_failed_only),
    )
    if warning_check:
        print(f"[summary] keypoint pass/fail CSV: {kp_passfail_csv}")

    if run_region_hypothesis:
        print("[pipeline] Step 3/4: Region detector hypothesis (batch outputs)")
        print(f"[pipeline] aircraft_pipeline_root={aircraft_pipeline_root}")
        print(
            f"[pipeline] region_hypothesis_visualize={bool(batch_region_hypothesis_visualize_outputs)} "
            f"max_scenes={int(region_hypothesis_vis_max_scenes)} "
            f"center_sphere_radius={float(region_hypothesis_center_sphere_radius):.3f}"
        )
        if region_hypothesis_python is not None:
            print(f"[pipeline] region_hypothesis_python={region_hypothesis_python}")
        try:
            if region_hypothesis_python is None:
                region_csv, region_total, region_pass = _run_region_hypothesis(
                    unique_scenes=unique_scenes_requested,
                    engine_region_root=engine_region_root,
                    out_root=out_root,
                    aircraft_pipeline_root=aircraft_pipeline_root,
                    min_engines=int(region_hypothesis_min_engines),
                    require_front_gear=bool(region_hypothesis_require_front_gear),
                    visualize_outputs=bool(batch_region_hypothesis_visualize_outputs),
                    vis_max_scenes=int(region_hypothesis_vis_max_scenes),
                    center_sphere_radius=float(region_hypothesis_center_sphere_radius),
                    split_windows=bool(region_hypothesis_split_windows),
                )
            else:
                region_csv, region_total, region_pass = _run_region_hypothesis_via_python(
                    python_exe=region_hypothesis_python,
                    unique_scenes=unique_scenes_requested,
                    engine_region_root=engine_region_root,
                    out_root=out_root,
                    aircraft_pipeline_root=aircraft_pipeline_root,
                    min_engines=int(region_hypothesis_min_engines),
                    require_front_gear=bool(region_hypothesis_require_front_gear),
                    visualize_outputs=bool(batch_region_hypothesis_visualize_outputs),
                    vis_max_scenes=int(region_hypothesis_vis_max_scenes),
                    center_sphere_radius=float(region_hypothesis_center_sphere_radius),
                    split_windows=bool(region_hypothesis_split_windows),
                )
            print(f"[summary] region hypothesis CSV: {region_csv}")
            print(
                f"[summary] region hypothesis: total={int(region_total)} "
                f"pass={int(region_pass)} fail={int(region_total - region_pass)} "
                f"min_engines={int(region_hypothesis_min_engines)} "
                f"require_front_gear={bool(region_hypothesis_require_front_gear)}"
            )
        except Exception as e:
            print(f"[warn] region hypothesis failed: {type(e).__name__}: {e}")

    if run_merger_pose:
        print("[pipeline] Step 4/4: Merger pose hypothesis (batch outputs)")
        if merger_pose_python is not None:
            print(f"[pipeline] merger_pose_python={merger_pose_python}")
        print(
            f"[pipeline] merger_pose_visualize={bool(batch_merger_pose_visualize)} "
            f"show_detected_parts={bool(merger_pose_show_detected_parts)}"
        )
        print(
            f"[pipeline] merger_pose_use_region_pcd_only="
            f"{bool(merger_pose_use_region_pcd_only)}"
        )
        try:
            merger_input_root = _build_merger_pose_input_dir(
                scene_pcd_paths=pcd_paths,
                out_root=out_root,
                use_region_pcd_only=bool(merger_pose_use_region_pcd_only),
                engine_region_root=(
                    engine_region_root
                    if bool(merger_pose_use_region_pcd_only)
                    else None
                ),
            )
            if merger_pose_python is None:
                raise RuntimeError("merger_pose_python is unresolved")
            merger_csv, merger_total, merger_pass = _run_merger_pose_via_python(
                python_exe=merger_pose_python,
                aircraft_pipeline_root=aircraft_pipeline_root,
                pcd_input_root=merger_input_root,
                out_root=out_root,
                detect_all_parts=bool(merger_pose_detect_all_parts),
                require_final_graph=bool(merger_pose_require_final_graph),
                visualize=bool(batch_merger_pose_visualize),
                show_detected_parts=bool(merger_pose_show_detected_parts),
                log_mode=str(merger_pose_log_mode),
            )
            print(f"[summary] merger pose CSV: {merger_csv}")
            print(
                f"[summary] merger pose: total={int(merger_total)} "
                f"pass={int(merger_pass)} fail={int(merger_total - merger_pass)} "
                f"detect_all_parts={bool(merger_pose_detect_all_parts)} "
                f"require_final_graph={bool(merger_pose_require_final_graph)}"
            )
            if save_merger_pose_fixed_csv:
                try:
                    _copy_merger_pose_csv_to_fixed_path(
                        src_csv=merger_csv,
                        dst_csv=merger_pose_fixed_csv_path,
                    )
                    print(
                        f"[summary] merger pose CSV fixed copy: "
                        f"{Path(merger_pose_fixed_csv_path).expanduser().resolve()}"
                    )
                except Exception as e:
                    print(
                        f"[warn] failed to copy merger pose CSV to fixed path "
                        f"({merger_pose_fixed_csv_path}): {type(e).__name__}: {e}"
                    )
        except Exception as e:
            print(f"[warn] merger pose failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
