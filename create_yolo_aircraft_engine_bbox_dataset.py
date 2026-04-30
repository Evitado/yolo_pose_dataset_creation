#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a YOLO detection dataset from an existing YOLO pose dataset.

Input dataset is expected to look like:
  <pose_dataset>/
    images/train|val|test/*.png
    labels/train|val|test/*.txt
    aircraft_pose.yaml  (contains keypoint names)

Output dataset:
  <out_dir>/
    images/train|val|test/*.png
    labels/train|val|test/*.txt
    aircraft_engine_det.yaml
    vis/train|val|test/*.png    (optional)

Per image, this exporter writes:
  - aircraft bbox (class 0), copied from pose bbox
  - engine-left bbox (class 1), centered at left engine keypoint
  - engine-right bbox (class 2), centered at right engine keypoint
  - front-gear bbox (class 3), centered at front-gear keypoint

Engine/front-gear bbox size is fixed in pixels and converted to normalized YOLO
xywh using the image resolution.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import cv2

from config_dataset import (
    OUT_DIR,
    ENGINE_LEFT_KP_NAME,
    ENGINE_RIGHT_KP_NAME,
    ENGINE_VIS_BBOX_HALF_W,
    ENGINE_VIS_BBOX_HALF_H,
    SYN_KP_NAME,
    NOSE_VIS_BBOX_HALF_W,
    NOSE_VIS_BBOX_HALF_H,
)

# Edit these paths directly if you want in-code defaults.
POSE_DATASET_PATH = str(Path(OUT_DIR).expanduser())
OUTPUT_DATASET_PATH = f"{POSE_DATASET_PATH}_det_aircraft_engine"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Convert a YOLO pose dataset into YOLO detection labels with "
            "aircraft + engine + front-gear boxes."
        )
    )
    ap.add_argument("--pose-dataset", type=str, default=POSE_DATASET_PATH, help="Input pose dataset root")
    ap.add_argument("--out-dir", type=str, default=OUTPUT_DATASET_PATH, help="Output detection dataset root")
    ap.add_argument("--left-kp-name", type=str, default=ENGINE_LEFT_KP_NAME, help="Left engine keypoint name")
    ap.add_argument("--right-kp-name", type=str, default=ENGINE_RIGHT_KP_NAME, help="Right engine keypoint name")
    ap.add_argument("--front-kp-name", type=str, default=SYN_KP_NAME, help="Front gear keypoint name")
    ap.add_argument(
        "--engine-half-w",
        type=int,
        default=int(ENGINE_VIS_BBOX_HALF_W),
        help="Half-width of engine bbox in pixels",
    )
    ap.add_argument(
        "--engine-half-h",
        type=int,
        default=int(ENGINE_VIS_BBOX_HALF_H),
        help="Half-height of engine bbox in pixels",
    )
    ap.add_argument(
        "--front-half-w",
        type=int,
        default=int(NOSE_VIS_BBOX_HALF_W),
        help="Half-width of front-gear bbox in pixels",
    )
    ap.add_argument(
        "--front-half-h",
        type=int,
        default=int(NOSE_VIS_BBOX_HALF_H),
        help="Half-height of front-gear bbox in pixels",
    )
    vis = ap.add_mutually_exclusive_group()
    vis.add_argument(
        "--make-viz",
        dest="make_viz",
        action="store_true",
        help="Save visualization images with aircraft/engine/front-gear bboxes under out_dir/vis/<split>/",
    )
    vis.add_argument(
        "--no-make-viz",
        dest="make_viz",
        action="store_false",
        help="Skip visualization image export.",
    )
    ap.set_defaults(make_viz=True)
    return ap.parse_args()


def _load_keypoint_names_from_pose_yaml(pose_yaml: Path) -> List[str]:
    if not pose_yaml.exists() or not pose_yaml.is_file():
        raise RuntimeError(f"Missing pose yaml: {pose_yaml}")

    lines = pose_yaml.read_text(encoding="utf-8", errors="ignore").splitlines()
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
    if not names:
        raise RuntimeError(f"No keypoints found in yaml: {pose_yaml}")
    return names


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _clip_xywhn(xc: float, yc: float, w: float, h: float) -> Optional[Tuple[float, float, float, float]]:
    x1 = _clamp01(xc - 0.5 * w)
    y1 = _clamp01(yc - 0.5 * h)
    x2 = _clamp01(xc + 0.5 * w)
    y2 = _clamp01(yc + 0.5 * h)
    if x2 <= x1 or y2 <= y1:
        return None
    xc2 = 0.5 * (x1 + x2)
    yc2 = 0.5 * (y1 + y2)
    w2 = x2 - x1
    h2 = y2 - y1
    return xc2, yc2, w2, h2


def _fmt_det_line(cls_id: int, xc: float, yc: float, w: float, h: float) -> str:
    return f"{int(cls_id)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


def _xywhn_to_xyxy_px(
    xc: float,
    yc: float,
    bw: float,
    bh: float,
    img_w: int,
    img_h: int,
) -> Optional[Tuple[int, int, int, int]]:
    x1 = int(round((float(xc) - 0.5 * float(bw)) * float(img_w)))
    y1 = int(round((float(yc) - 0.5 * float(bh)) * float(img_h)))
    x2 = int(round((float(xc) + 0.5 * float(bw)) * float(img_w)))
    y2 = int(round((float(yc) + 0.5 * float(bh)) * float(img_h)))

    x1 = max(0, min(img_w - 1, x1))
    y1 = max(0, min(img_h - 1, y1))
    x2 = max(0, min(img_w - 1, x2))
    y2 = max(0, min(img_h - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _parse_pose_line(line: str, kpt_count: int) -> Optional[Tuple[float, float, float, float, List[Tuple[float, float, float]]]]:
    s = str(line).strip()
    if not s:
        return None
    parts = s.split()
    needed = 5 + 3 * int(kpt_count)
    if len(parts) < needed:
        return None

    try:
        # class id is parts[0], unused for conversion
        cx = float(parts[1])
        cy = float(parts[2])
        bw = float(parts[3])
        bh = float(parts[4])
    except Exception:
        return None

    kps: List[Tuple[float, float, float]] = []
    for i in range(kpt_count):
        j = 5 + 3 * i
        try:
            x = float(parts[j])
            y = float(parts[j + 1])
            v = float(parts[j + 2])
        except Exception:
            return None
        kps.append((x, y, v))
    return cx, cy, bw, bh, kps


def _ensure_split_dirs(out_root: Path, make_viz: bool) -> None:
    for rel in (
        "images/train",
        "images/val",
        "images/test",
        "labels/train",
        "labels/val",
        "labels/test",
    ):
        (out_root / rel).mkdir(parents=True, exist_ok=True)
    if make_viz:
        for rel in ("vis/train", "vis/val", "vis/test"):
            (out_root / rel).mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()

    pose_root = Path(args.pose_dataset).expanduser().resolve()
    out_root = Path(args.out_dir).expanduser().resolve()
    pose_yaml = pose_root / "aircraft_pose.yaml"
    keypoint_names = _load_keypoint_names_from_pose_yaml(pose_yaml)

    left_name = str(args.left_kp_name).strip()
    right_name = str(args.right_kp_name).strip()
    front_name = str(args.front_kp_name).strip()
    if left_name not in keypoint_names:
        raise RuntimeError(
            f"Left engine keypoint '{left_name}' not found in keypoints: {keypoint_names}"
        )
    if right_name not in keypoint_names:
        raise RuntimeError(
            f"Right engine keypoint '{right_name}' not found in keypoints: {keypoint_names}"
        )
    if front_name not in keypoint_names:
        raise RuntimeError(
            f"Front gear keypoint '{front_name}' not found in keypoints: {keypoint_names}"
        )

    left_idx = keypoint_names.index(left_name)
    right_idx = keypoint_names.index(right_name)
    front_idx = keypoint_names.index(front_name)

    half_w_px = int(args.engine_half_w)
    half_h_px = int(args.engine_half_h)
    front_half_w_px = int(args.front_half_w)
    front_half_h_px = int(args.front_half_h)
    if half_w_px <= 0 or half_h_px <= 0:
        raise RuntimeError("--engine-half-w and --engine-half-h must be > 0")
    if front_half_w_px <= 0 or front_half_h_px <= 0:
        raise RuntimeError("--front-half-w and --front-half-h must be > 0")

    make_viz = bool(args.make_viz)
    _ensure_split_dirs(out_root, make_viz=make_viz)

    img_count = 0
    lbl_count = 0
    aircraft_count = 0
    engine_left_count = 0
    engine_right_count = 0
    front_gear_count = 0
    viz_count = 0
    skipped_missing_label = 0
    skipped_bad_image = 0

    for split in ("train", "val", "test"):
        src_img_dir = pose_root / "images" / split
        src_lbl_dir = pose_root / "labels" / split
        dst_img_dir = out_root / "images" / split
        dst_lbl_dir = out_root / "labels" / split
        dst_vis_dir = out_root / "vis" / split

        if not src_img_dir.exists():
            continue

        images = sorted(
            [
                p
                for p in src_img_dir.iterdir()
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
            ]
        )
        for img_path in images:
            stem = img_path.stem
            src_lbl = src_lbl_dir / f"{stem}.txt"
            if not src_lbl.exists() or not src_lbl.is_file():
                skipped_missing_label += 1
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                skipped_bad_image += 1
                continue
            h, w = img.shape[:2]
            if h <= 0 or w <= 0:
                skipped_bad_image += 1
                continue

            det_lines: List[str] = []
            det_boxes: List[Tuple[int, float, float, float, float]] = []
            raw_lines = src_lbl.read_text(encoding="utf-8", errors="ignore").splitlines()
            for ln in raw_lines:
                parsed = _parse_pose_line(ln, kpt_count=len(keypoint_names))
                if parsed is None:
                    continue
                cx, cy, bw, bh, kps = parsed

                # 0: aircraft bbox from pose label
                box_air = _clip_xywhn(cx, cy, bw, bh)
                if box_air is not None:
                    det_lines.append(_fmt_det_line(0, *box_air))
                    det_boxes.append((0, *box_air))
                    aircraft_count += 1

                # engine/front-gear boxes from keypoints (only when visible)
                for cls_id, kp_idx, kp_half_w, kp_half_h in (
                    (1, left_idx, half_w_px, half_h_px),
                    (2, right_idx, half_w_px, half_h_px),
                    (3, front_idx, front_half_w_px, front_half_h_px),
                ):
                    xk, yk, vk = kps[kp_idx]
                    if vk <= 0:
                        continue
                    bw_e = (2.0 * kp_half_w) / float(w)
                    bh_e = (2.0 * kp_half_h) / float(h)
                    box_eng = _clip_xywhn(xk, yk, bw_e, bh_e)
                    if box_eng is None:
                        continue
                    det_lines.append(_fmt_det_line(cls_id, *box_eng))
                    det_boxes.append((cls_id, *box_eng))
                    if cls_id == 1:
                        engine_left_count += 1
                    elif cls_id == 2:
                        engine_right_count += 1
                    else:
                        front_gear_count += 1

            # Keep image and label in sync: copy image even if zero detections.
            shutil.copy2(img_path, dst_img_dir / img_path.name)
            img_count += 1

            (dst_lbl_dir / f"{stem}.txt").write_text(
                ("\n".join(det_lines) + "\n") if det_lines else "",
                encoding="utf-8",
            )
            lbl_count += 1

            if make_viz:
                vis_img = img.copy()
                for cls_id, xc, yc, bw, bh in det_boxes:
                    rect = _xywhn_to_xyxy_px(xc, yc, bw, bh, img_w=w, img_h=h)
                    if rect is None:
                        continue
                    x1, y1, x2, y2 = rect
                    if int(cls_id) == 0:
                        color = (0, 255, 0)
                        name = "aircraft"
                    elif int(cls_id) == 1:
                        color = (255, 200, 0)
                        name = "engine_left"
                    elif int(cls_id) == 2:
                        color = (0, 200, 255)
                        name = "engine_right"
                    else:
                        color = (0, 140, 255)
                        name = "front_gear"
                    cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        vis_img,
                        name,
                        (x1 + 2, max(10, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        color,
                        1,
                        cv2.LINE_AA,
                    )
                cv2.imwrite(str(dst_vis_dir / img_path.name), vis_img)
                viz_count += 1

    yaml_text = (
        "# YOLO detection dataset — aircraft + engine + front-gear boxes\n"
        f"path: {out_root}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "names:\n"
        "  0: aircraft\n"
        "  1: engine_left\n"
        "  2: engine_right\n"
        "  3: front_gear\n"
    )
    (out_root / "aircraft_engine_det.yaml").write_text(yaml_text, encoding="utf-8")

    print(f"[done] output: {out_root}")
    print(f"[done] yaml:   {out_root / 'aircraft_engine_det.yaml'}")
    if make_viz:
        print(f"[done] vis:    {out_root / 'vis'}")
    print(
        f"[summary] images={img_count} labels={lbl_count} "
        f"aircraft_boxes={aircraft_count} "
        f"engine_left_boxes={engine_left_count} engine_right_boxes={engine_right_count} "
        f"front_gear_boxes={front_gear_count}"
    )
    if make_viz:
        print(f"[summary] vis_images={viz_count}")
    print(
        f"[summary] skipped_missing_label={skipped_missing_label} "
        f"skipped_bad_image={skipped_bad_image}"
    )


if __name__ == "__main__":
    main()
