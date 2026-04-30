#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Image pipeline:
1) Run YOLO pose on one test image (or all images in a directory), mapped to H5 scene by filename stem
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
from pathlib import Path


# =========================
# Code-level toggles (edit here)
# =========================
VISUALIZE_PCD: bool = False
RUN_WARNING_CHECK: bool = True
RUN_WARNING_PASS_FAIL: bool = True
USE_SCENE_H5_TRANSFORM: bool = True
SHOW_AXES: bool = False

# PCD/keypoint rendering/check settings
KPT_COUNT: int = 3
KPT_RADIUS_M: float = 0.25
WARNING_BOX_SCALE: float = 1.0
# Show visualization only for scenes with at least this many warning-keypoint FAILs.
# Set 0 to disable this filter.
MIN_WARNING_FAIL_KP_TO_VISUALIZE: int = 2
WARNING_FALLBACK_KP_NAMES: str = "front_wheels_mid,engine_left_box_center,engine_right_box_center"

# YOLO run settings
SAVE_DEBUG_IMAGE: bool = True
IMG_SIZE: int = 1024
YOLO_CONF: float = 0.05
DEVICE: str = "0"
KP_CONF_THR: float = 0.2
WARNING_CONF_THR: float = 0.95
KP_PATCH_RADIUS: int = 3
CHECK_BBOX_COVERAGE: bool = True
BBOX_FULL_THR: float = 0.7

# Default paths (can still be overridden by CLI args)
DEFAULT_IMAGE_PATH: str = "/home/femi/yolo_pose_dataset_creation/aircraft_pose_with_normalising_applied_multifield_only_3_2/images/test"
DEFAULT_WEIGHTS_PATH: str = "/home/femi/Thesis_data/yolo_data/yolo_keypoint/yolo_train/aircraft_pose_rect_1024x128_y26m-3/weights/best.pt"
DEFAULT_SOURCE_H5_ROOT: str = "/home/femi/Benchmarking_framework/Data/warning_b_test_h5"
DEFAULT_OUT_DIR: str = "/home/femi/yolo_pose_dataset_creation/pcd_from_yolo"
DEFAULT_YAML_KP_NAMES: str = (
    "/home/femi/yolo_pose_dataset_creation/"
    "aircraft_pose_with_normalising_applied_multifield_only_3_2/aircraft_pose.yaml"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run YOLO on one test image (or image directory), export PCD, then optional visualization and warning-box checks."
        )
    )
    p.add_argument("--image-path", type=str, default=DEFAULT_IMAGE_PATH, help="Single test image path (or directory path)")
    p.add_argument("--image-dir", type=str, default="", help="Directory containing test images")
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

    image_path_raw = str(args.image_path or "").strip()
    image_dir_raw = str(args.image_dir or "").strip()
    weights = str(args.weights or "").strip()
    source = str(args.source or "").strip()
    out_dir = str(args.out or "").strip()
    yaml_kp_names = str(args.yaml_kp_names or "").strip()

    if not weights:
        raise RuntimeError("Please set --weights (or DEFAULT_WEIGHTS_PATH in code).")
    if not source:
        raise RuntimeError("Please set --source (or DEFAULT_SOURCE_H5_ROOT in code).")
    if not out_dir:
        raise RuntimeError("Please set --out (or DEFAULT_OUT_DIR in code).")
    if image_path_raw and image_dir_raw:
        raise RuntimeError("Use only one of --image-path or --image-dir.")

    image_paths: list[Path] = []
    run_image_path: str | None = None
    run_image_dir: str | None = None

    if image_dir_raw:
        image_dir = Path(image_dir_raw).expanduser().resolve()
        image_paths = [p.resolve() for p in yolo_pcd._collect_images_from_dir(image_dir)]
        run_image_dir = str(image_dir)
    elif image_path_raw:
        p = Path(image_path_raw).expanduser().resolve()
        if p.exists() and p.is_dir():
            image_paths = [pp.resolve() for pp in yolo_pcd._collect_images_from_dir(p)]
            run_image_dir = str(p)
        else:
            resolved_image = yolo_pcd._resolve_image_path_with_split_fallback(image_path_raw)
            image_paths = [resolved_image.resolve()]
            run_image_path = str(resolved_image)
    else:
        raise RuntimeError("Please set --image-path or --image-dir.")

    unique_scenes = [str(p.stem) for p in image_paths]

    out_root = Path(out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    kp_conf_csv_path = out_root / "keypoint_confidence.csv"
    bbox_cov_csv_path = out_root / "bbox_aircraft_coverage.csv"
    kp_passfail_csv_path = out_root / "keypoint_pass_fail_confidence.csv"

    print("[pipeline] Step 1/2: YOLO -> PCD export")
    if run_image_path:
        print(f"[pipeline] image: {run_image_path}")
    else:
        print(f"[pipeline] image-dir: {run_image_dir} (count={len(image_paths)})")
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
        image_path=run_image_path,
        image_dir=run_image_dir,
        print_kp_conf=True,
        kp_conf_csv=str(kp_conf_csv_path),
        check_bbox_coverage=bool(CHECK_BBOX_COVERAGE),
        bbox_full_thr=float(BBOX_FULL_THR),
        bbox_cov_csv=(str(bbox_cov_csv_path) if bool(CHECK_BBOX_COVERAGE) else ""),
    )
    if bool(CHECK_BBOX_COVERAGE):
        print(f"[pipeline] bbox coverage csv: {bbox_cov_csv_path}")
        bbox_by_scene: dict[str, dict[str, str]] = {}
        if bbox_cov_csv_path.exists() and bbox_cov_csv_path.is_file():
            with bbox_cov_csv_path.open("r", encoding="utf-8", newline="") as f_csv:
                for row in csv.DictReader(f_csv):
                    key = str(row.get("unique_scene", "")).strip()
                    if key:
                        bbox_by_scene[key] = row
        else:
            print("[pipeline] bbox pass/fail CSV not found after run.")
        if bbox_by_scene:
            print("[pipeline] BBOX PASS/FAIL per scene:")
            pass_n = 0
            fail_n = 0
            unknown_n = 0
            for scene in unique_scenes:
                row = bbox_by_scene.get(scene)
                if row is None:
                    unknown_n += 1
                    print(f"  [bbox-passfail] {scene}: status=UNKNOWN reason=missing_csv_row")
                    continue
                status = str(row.get("bbox_status", "UNKNOWN")).strip().upper() or "UNKNOWN"
                reason = str(row.get("bbox_reason", "")).strip() or "n/a"
                x1 = str(row.get("bbox_x1", "")).strip()
                y1 = str(row.get("bbox_y1", "")).strip()
                x2 = str(row.get("bbox_x2", "")).strip()
                y2 = str(row.get("bbox_y2", "")).strip()
                rec = str(row.get("aircraft_recall", "")).strip()
                if status == "PASS":
                    pass_n += 1
                elif status == "FAIL":
                    fail_n += 1
                else:
                    unknown_n += 1
                bbox_txt = f"({x1},{y1},{x2},{y2})" if all([x1, y1, x2, y2]) else "(n/a)"
                rec_txt = rec if rec else "n/a"
                print(
                    f"  [bbox-passfail] {scene}: status={status} "
                    f"reason={reason} bbox={bbox_txt} recall={rec_txt}"
                )
            print(
                f"[summary] bbox pass/fail (from CSV): total={len(unique_scenes)} "
                f"pass={pass_n} fail={fail_n} unknown={unknown_n}"
            )

    pcd_paths = [out_root / f"{scene}.pcd" for scene in unique_scenes]
    pcd_paths = [p for p in pcd_paths if p.exists() and p.is_file()]
    if not pcd_paths:
        raise RuntimeError(
            f"No expected output PCD files found under: {out_root}. "
            f"Requested scenes: {len(unique_scenes)}"
        )
    if len(pcd_paths) < len(unique_scenes):
        print(
            f"[pipeline] Warning: only {len(pcd_paths)}/{len(unique_scenes)} requested scenes "
            "produced PCD files."
        )

    print("[pipeline] Step 2/2: PCD view/check")
    print(f"[pipeline] code-toggle VISUALIZE_PCD={bool(VISUALIZE_PCD)}")
    print(f"[pipeline] code-toggle RUN_WARNING_CHECK={bool(RUN_WARNING_CHECK)}")
    print(f"[pipeline] code-toggle RUN_WARNING_PASS_FAIL={bool(RUN_WARNING_PASS_FAIL)}")
    print(
        "[pipeline] code-toggle MIN_WARNING_FAIL_KP_TO_VISUALIZE="
        f"{int(max(0, MIN_WARNING_FAIL_KP_TO_VISUALIZE))}"
    )

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
        warning_conf_threshold=float(WARNING_CONF_THR),
        min_warning_fail_kp_to_visualize=int(max(0, MIN_WARNING_FAIL_KP_TO_VISUALIZE)),
    )
    if bool(RUN_WARNING_CHECK):
        print(f"[pipeline] keypoint pass/fail confidence csv: {kp_passfail_csv_path}")


if __name__ == "__main__":
    main()
