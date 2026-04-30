#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Edit YOLO detection bboxes (aircraft/engine/front_gear) and export a new dataset copy.

Expected input dataset layout:
  <dataset_root>/
    images/train|val|test/*
    labels/train|val|test/*.txt
    aircraft_engine_det.yaml

Output layout:
  <out_dir>/
    images/train|val|test/*
    labels/train|val|test/*.txt
    aircraft_engine_det.yaml
    vis/train|val|test/*.png     (optional)

BBox edits are applied in normalized YOLO xywh:
  - width/height scaling
  - center x/y shifting

You can apply:
  - global edits to all classes
  - class-specific edits for aircraft, engine_left, engine_right
  - optional interactive suggestion of missing front_gear boxes
"""

from __future__ import annotations

import argparse
import ast
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2

from config_dataset import OUT_DIR


# Code-level path overrides (set True to ignore CLI paths below)
USE_CODE_PATH_OVERRIDES: bool = True
CODE_DATASET_ROOT: str = "/home/femi/yolo_pose_dataset_creation/aircraft_engine_det_with_front_edited"
CODE_OUT_DIR: str = "/home/femi/yolo_pose_dataset_creation/aircraft_engine_det_with_front_edited3"
CODE_YAML_NAME: str = "aircraft_engine_det.yaml"

DEFAULT_SUGGEST_FRONT_GEAR_STEM_CONTAINS: str = "a380_800"
DEFAULT_FRONT_GEAR_DX_OVER_ENGINE_SPAN: float = 0.184
DEFAULT_FRONT_GEAR_DY_OVER_ENGINE_H: float = 0.484
DEFAULT_FRONT_GEAR_W_OVER_ENGINE_W: float = 0.8333226667
DEFAULT_FRONT_GEAR_H_OVER_ENGINE_H: float = 0.875
DEFAULT_SUGGEST_ENGINE_STEM_CONTAINS: str = ""


def parse_args() -> argparse.Namespace:
    if USE_CODE_PATH_OVERRIDES:
        default_in = Path(CODE_DATASET_ROOT).expanduser()
        default_out = Path(CODE_OUT_DIR).expanduser()
        default_yaml = str(CODE_YAML_NAME).strip() or "aircraft_engine_det.yaml"
    else:
        default_in = Path("/home/femi/yolo_pose_dataset_creation/aircraft_engine_det_test")
        if not default_in.exists():
            default_in = Path(f"{str(Path(OUT_DIR).expanduser())}_det_aircraft_engine")
        default_out = Path(f"{str(default_in)}_edited")
        default_yaml = "aircraft_engine_det.yaml"

    ap = argparse.ArgumentParser(
        description="Edit YOLO detection bboxes and export edited labels/images."
    )
    ap.add_argument("--dataset-root", type=str, default=str(default_in), help="Input detection dataset root")
    ap.add_argument("--out-dir", type=str, default=str(default_out), help="Output edited dataset root")
    ap.add_argument(
        "--yaml-name",
        type=str,
        default=default_yaml,
        help="YAML filename inside dataset root",
    )

    # Global edits
    ap.add_argument("--scale-all", type=float, default=1.0, help="Scale w/h for all classes")
    ap.add_argument("--shift-x-all", type=float, default=0.0, help="Shift x-center for all classes")
    ap.add_argument("--shift-y-all", type=float, default=0.0, help="Shift y-center for all classes")

    # Aircraft edits
    ap.add_argument("--scale-aircraft", type=float, default=1.0, help="Extra scale for aircraft")
    ap.add_argument("--shift-x-aircraft", type=float, default=0.0, help="Extra x shift for aircraft")
    ap.add_argument("--shift-y-aircraft", type=float, default=0.0, help="Extra y shift for aircraft")

    # Engine-left edits
    ap.add_argument("--scale-engine-left", type=float, default=1.0, help="Extra scale for engine_left")
    ap.add_argument("--shift-x-engine-left", type=float, default=0.0, help="Extra x shift for engine_left")
    ap.add_argument("--shift-y-engine-left", type=float, default=0.0, help="Extra y shift for engine_left")

    # Engine-right edits
    ap.add_argument("--scale-engine-right", type=float, default=1.0, help="Extra scale for engine_right")
    ap.add_argument("--shift-x-engine-right", type=float, default=0.0, help="Extra x shift for engine_right")
    ap.add_argument("--shift-y-engine-right", type=float, default=0.0, help="Extra y shift for engine_right")

    vis = ap.add_mutually_exclusive_group()
    vis.add_argument("--make-viz", dest="make_viz", action="store_true", help="Save vis images to out_dir/vis")
    vis.add_argument("--no-make-viz", dest="make_viz", action="store_false", help="Skip vis image export")
    interactive_mode = ap.add_mutually_exclusive_group()
    interactive_mode.add_argument(
        "--interactive",
        dest="interactive",
        action="store_true",
        help=(
            "Interactive bbox editor (mouse drag/resize). "
            "Controls: left-drag box/corner, tab cycle boxes, x toggle keep/drop, "
            "s save, n next, q quit."
        ),
    )
    interactive_mode.add_argument(
        "--no-interactive",
        dest="interactive",
        action="store_false",
        help="Run non-interactive batch edit mode.",
    )
    suggest_front_group = ap.add_mutually_exclusive_group()
    suggest_front_group.add_argument(
        "--suggest-missing-front-gear",
        dest="suggest_missing_front_gear",
        action="store_true",
        help=(
            "Interactive mode only: when engine_left and engine_right exist but front_gear is missing, "
            "add a suggested front_gear bbox that you can keep/drop."
        ),
    )
    suggest_front_group.add_argument(
        "--no-suggest-missing-front-gear",
        dest="suggest_missing_front_gear",
        action="store_false",
        help="Disable automatic front_gear suggestion boxes in interactive mode.",
    )
    suggest_front_keep_group = ap.add_mutually_exclusive_group()
    suggest_front_keep_group.add_argument(
        "--suggest-front-gear-keep",
        dest="suggest_front_gear_keep",
        action="store_true",
        help="Suggested front_gear boxes start as enabled.",
    )
    suggest_front_keep_group.add_argument(
        "--suggest-front-gear-off",
        dest="suggest_front_gear_keep",
        action="store_false",
        help="Suggested front_gear boxes start as disabled (default).",
    )
    suggest_engine_group = ap.add_mutually_exclusive_group()
    suggest_engine_group.add_argument(
        "--suggest-missing-engines",
        dest="suggest_missing_engines",
        action="store_true",
        help=(
            "Interactive mode only: when one of engine_left/engine_right is missing, "
            "add a suggested mirrored bbox for the missing side."
        ),
    )
    suggest_engine_group.add_argument(
        "--no-suggest-missing-engines",
        dest="suggest_missing_engines",
        action="store_false",
        help="Disable automatic engine_left/engine_right suggestion boxes in interactive mode.",
    )
    suggest_engine_keep_group = ap.add_mutually_exclusive_group()
    suggest_engine_keep_group.add_argument(
        "--suggest-engine-keep",
        dest="suggest_engine_keep",
        action="store_true",
        help="Suggested engine boxes start as enabled.",
    )
    suggest_engine_keep_group.add_argument(
        "--suggest-engine-off",
        dest="suggest_engine_keep",
        action="store_false",
        help="Suggested engine boxes start as disabled (default).",
    )
    ap.set_defaults(
        make_viz=True,
        interactive=True,
        suggest_missing_front_gear=True,
        suggest_front_gear_keep=False,
        suggest_missing_engines=True,
        suggest_engine_keep=False,
    )
    ap.add_argument(
        "--split",
        type=str,
        default="valnnnnn",
        choices=["all", "train", "val", "test"],
        help="Dataset split to process (default: all).",
    )
    ap.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start image index after sorting.",
    )
    ap.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Maximum images to process (0 = all).",
    )
    ap.add_argument(
        "--stem-contains",
        type=str,
        default="",
        help=(
            "Optional image stem substring filter used by fixed-size overrides "
            "(example: '757_300'). Empty = apply override logic to all stems."
        ),
    )
    ap.add_argument(
        "--engine-fixed-w-px",
        type=float,
        default=0.0,
        help=(
            "If >0 and --engine-fixed-h-px>0, force engine_left/right bbox width "
            "to this many pixels (after center shifts)."
        ),
    )
    ap.add_argument(
        "--engine-fixed-h-px",
        type=float,
        default=0.0,
        help=(
            "If >0 and --engine-fixed-w-px>0, force engine_left/right bbox height "
            "to this many pixels (after center shifts)."
        ),
    )
    ap.add_argument(
        "--suggest-front-gear-stem-contains",
        type=str,
        default=DEFAULT_SUGGEST_FRONT_GEAR_STEM_CONTAINS,
        help=(
            "Only suggest missing front_gear for images with this substring in stem. "
            "Set empty string to allow all stems."
        ),
    )
    ap.add_argument(
        "--suggest-front-gear-dx-over-engine-span",
        type=float,
        default=DEFAULT_FRONT_GEAR_DX_OVER_ENGINE_SPAN,
        help=(
            "Suggested front_gear x offset from engine midpoint, normalized by engine-center span."
        ),
    )
    ap.add_argument(
        "--suggest-front-gear-dy-over-engine-h",
        type=float,
        default=DEFAULT_FRONT_GEAR_DY_OVER_ENGINE_H,
        help=(
            "Suggested front_gear y offset from engine midpoint, normalized by average engine height."
        ),
    )
    ap.add_argument(
        "--suggest-front-gear-w-over-engine-w",
        type=float,
        default=DEFAULT_FRONT_GEAR_W_OVER_ENGINE_W,
        help="Suggested front_gear width, normalized by average engine width.",
    )
    ap.add_argument(
        "--suggest-front-gear-h-over-engine-h",
        type=float,
        default=DEFAULT_FRONT_GEAR_H_OVER_ENGINE_H,
        help="Suggested front_gear height, normalized by average engine height.",
    )
    ap.add_argument(
        "--suggest-engine-stem-contains",
        type=str,
        default=DEFAULT_SUGGEST_ENGINE_STEM_CONTAINS,
        help=(
            "Only suggest missing engine_left/engine_right for images with this substring in stem. "
            "Empty string = allow all stems."
        ),
    )

    args = ap.parse_args()
    if USE_CODE_PATH_OVERRIDES:
        args.dataset_root = str(default_in)
        args.out_dir = str(default_out)
        args.yaml_name = str(default_yaml)
    return args


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _clip_xywhn(xc: float, yc: float, bw: float, bh: float) -> Optional[Tuple[float, float, float, float]]:
    x1 = _clamp01(float(xc) - 0.5 * float(bw))
    y1 = _clamp01(float(yc) - 0.5 * float(bh))
    x2 = _clamp01(float(xc) + 0.5 * float(bw))
    y2 = _clamp01(float(yc) + 0.5 * float(bh))
    if x2 <= x1 or y2 <= y1:
        return None
    return 0.5 * (x1 + x2), 0.5 * (y1 + y2), (x2 - x1), (y2 - y1)


def _fmt_line(cls_id: int, xc: float, yc: float, bw: float, bh: float) -> str:
    return f"{int(cls_id)} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def _xywhn_to_xyxy_px(
    xc: float, yc: float, bw: float, bh: float, img_w: int, img_h: int
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


def _xywhn_to_xyxy_float(xc: float, yc: float, bw: float, bh: float, img_w: int, img_h: int) -> Optional[List[float]]:
    w = float(bw) * float(img_w)
    h = float(bh) * float(img_h)
    x1 = float(xc) * float(img_w) - 0.5 * w
    y1 = float(yc) * float(img_h) - 0.5 * h
    x2 = float(xc) * float(img_w) + 0.5 * w
    y2 = float(yc) * float(img_h) + 0.5 * h
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _normalize_bbox_xyxy(box: List[float], img_w: int, img_h: int, min_size_px: float = 2.0) -> Optional[List[float]]:
    x1, y1, x2, y2 = [float(v) for v in box]
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    x1 = max(0.0, min(float(img_w - 1), x1))
    x2 = max(0.0, min(float(img_w - 1), x2))
    y1 = max(0.0, min(float(img_h - 1), y1))
    y2 = max(0.0, min(float(img_h - 1), y2))

    if x2 - x1 < float(min_size_px):
        cx = 0.5 * (x1 + x2)
        hw = 0.5 * float(min_size_px)
        x1 = max(0.0, cx - hw)
        x2 = min(float(img_w - 1), cx + hw)
    if y2 - y1 < float(min_size_px):
        cy = 0.5 * (y1 + y2)
        hh = 0.5 * float(min_size_px)
        y1 = max(0.0, cy - hh)
        y2 = min(float(img_h - 1), cy + hh)

    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _xyxy_float_to_xywhn(box: List[float], img_w: int, img_h: int) -> Optional[Tuple[float, float, float, float]]:
    x1, y1, x2, y2 = [float(v) for v in box]
    bw = max(0.0, x2 - x1) / float(img_w)
    bh = max(0.0, y2 - y1) / float(img_h)
    xc = ((x1 + x2) * 0.5) / float(img_w)
    yc = ((y1 + y2) * 0.5) / float(img_h)
    return _clip_xywhn(xc, yc, bw, bh)


def _pick_bbox_handle(box: List[float], x: float, y: float, radius: float) -> Optional[str]:
    x1, y1, x2, y2 = [float(v) for v in box]
    corners = {
        "tl": (x1, y1),
        "tr": (x2, y1),
        "br": (x2, y2),
        "bl": (x1, y2),
    }
    r2 = float(radius) * float(radius)
    for name, (cx, cy) in corners.items():
        if (x - cx) ** 2 + (y - cy) ** 2 <= r2:
            return name
    if x1 <= x <= x2 and y1 <= y <= y2:
        return "move"
    return None


def _apply_bbox_drag(start_box: List[float], handle: str, anchor: Tuple[float, float], cur: Tuple[float, float]) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in start_box]
    ax, ay = [float(v) for v in anchor]
    cx, cy = [float(v) for v in cur]
    dx, dy = cx - ax, cy - ay

    if handle == "move":
        return [x1 + dx, y1 + dy, x2 + dx, y2 + dy]
    if handle == "tl":
        return [cx, cy, x2, y2]
    if handle == "tr":
        return [x1, cy, cx, y2]
    if handle == "br":
        return [x1, y1, cx, cy]
    if handle == "bl":
        return [cx, y1, x2, cy]
    return [x1, y1, x2, y2]


def _draw_boxes_canvas(
    image: "cv2.typing.MatLike",
    boxes_xyxy: List[List[float]],
    cls_ids: List[int],
    keep_mask: List[bool],
    class_map: Dict[int, str],
    suggested_mask: Optional[List[bool]] = None,
    selected_idx: Optional[int] = None,
    show_handles: bool = False,
    title: str = "",
) -> "cv2.typing.MatLike":
    canvas = image.copy()

    for i, box in enumerate(boxes_xyxy):
        rect = _normalize_bbox_xyxy(list(box), image.shape[1], image.shape[0], min_size_px=1.0)
        if rect is None:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in rect]
        color, name = _class_color_name(int(cls_ids[i]), class_map)
        is_suggested = bool(suggested_mask[i]) if suggested_mask is not None and i < len(suggested_mask) else False
        if not bool(keep_mask[i]):
            if is_suggested:
                color = (
                    int(0.45 * float(color[0]) + 80.0),
                    int(0.45 * float(color[1]) + 80.0),
                    int(0.45 * float(color[2]) + 80.0),
                )
            else:
                color = (110, 110, 110)
        thickness = 3 if (selected_idx is not None and i == selected_idx) else 2
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        tag = f"{name}#{i}"
        if is_suggested:
            tag += " [suggest]"
        if not bool(keep_mask[i]):
            tag += " [off]"
        cv2.putText(
            canvas,
            tag,
            (x1 + 2, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

        if show_handles and selected_idx is not None and i == selected_idx and bool(keep_mask[i]):
            pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            for px, py in pts:
                cv2.circle(canvas, (int(px), int(py)), 4, (255, 255, 255), -1, lineType=cv2.LINE_AA)
                cv2.circle(canvas, (int(px), int(py)), 4, (0, 0, 0), 1, lineType=cv2.LINE_AA)

    if title:
        cv2.putText(
            canvas,
            title,
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    controls = "Mouse: drag box/corner | right-click/x: toggle keep | tab: next | r: reset | s: save | n: next | q: quit"
    cv2.putText(
        canvas,
        controls,
        (8, max(20, image.shape[0] - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _write_det_labels_from_xyxy(
    out_label_path: Path,
    boxes_xyxy: List[List[float]],
    cls_ids: List[int],
    keep_mask: List[bool],
    img_w: int,
    img_h: int,
) -> int:
    lines: List[str] = []
    kept = 0
    for i, box in enumerate(boxes_xyxy):
        if not bool(keep_mask[i]):
            continue
        norm = _normalize_bbox_xyxy(list(box), img_w, img_h, min_size_px=1.0)
        if norm is None:
            continue
        xywhn = _xyxy_float_to_xywhn(norm, img_w=img_w, img_h=img_h)
        if xywhn is None:
            continue
        lines.append(_fmt_line(int(cls_ids[i]), *xywhn))
        kept += 1
    out_label_path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
    return kept

def _parse_det_line(line: str) -> Optional[Tuple[int, float, float, float, float]]:
    s = str(line).strip()
    if not s:
        return None
    parts = s.split()
    if len(parts) < 5:
        return None
    try:
        cls_id = int(float(parts[0]))
        xc = float(parts[1])
        yc = float(parts[2])
        bw = float(parts[3])
        bh = float(parts[4])
    except Exception:
        return None
    return cls_id, xc, yc, bw, bh


def _normalize_class_token(name: str) -> str:
    tok = str(name or "").strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in tok:
        tok = tok.replace("__", "_")
    return tok


def _find_class_id_by_aliases(class_map: Dict[int, str], aliases: List[str]) -> Optional[int]:
    alias_set = {_normalize_class_token(a) for a in aliases}
    for cls_id in sorted(class_map.keys()):
        nm = _normalize_class_token(str(class_map[cls_id]))
        if nm in alias_set:
            return int(cls_id)
    return None


def _engine_side_for_class(cls_id: int, class_map: Dict[int, str]) -> Optional[str]:
    nm = _normalize_class_token(str(class_map.get(int(cls_id), "")))
    if int(cls_id) == 1 or nm in {"engine_left", "left_engine"}:
        return "left"
    if int(cls_id) == 2 or nm in {"engine_right", "right_engine"}:
        return "right"
    return None


def _find_aircraft_class_id(class_map: Dict[int, str]) -> Optional[int]:
    cls_id = _find_class_id_by_aliases(class_map, aliases=["aircraft"])
    if cls_id is not None:
        return cls_id
    if 0 in class_map:
        return 0
    return None


def _find_engine_class_ids(class_map: Dict[int, str]) -> Tuple[Optional[int], Optional[int]]:
    left_id = _find_class_id_by_aliases(class_map, aliases=["engine_left", "left_engine"])
    right_id = _find_class_id_by_aliases(class_map, aliases=["engine_right", "right_engine"])
    if left_id is None and 1 in class_map:
        left_id = 1
    if right_id is None and 2 in class_map:
        right_id = 2
    return left_id, right_id


def _find_largest_kept_box_index_for_class(
    cls_ids: List[int],
    boxes_xyxy: List[List[float]],
    keep_mask: List[bool],
    target_cls_id: Optional[int],
) -> Optional[int]:
    if target_cls_id is None:
        return None
    best_idx: Optional[int] = None
    best_area = -1.0
    for i, cls_id in enumerate(cls_ids):
        if int(cls_id) != int(target_cls_id):
            continue
        if not bool(keep_mask[i]):
            continue
        box = boxes_xyxy[i]
        area = max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))
        if area > best_area:
            best_area = area
            best_idx = i
    return best_idx


def _find_front_gear_class_id(class_map: Dict[int, str]) -> Optional[int]:
    cls_id = _find_class_id_by_aliases(
        class_map,
        aliases=[
            "front_gear",
            "frontgear",
            "frontwheel",
            "front_wheel",
            "front_wheels",
            "front_landing_gear",
            "nose_gear",
            "nose_wheel",
        ],
    )
    if cls_id is not None:
        return cls_id
    if 3 in class_map:
        return 3
    return None


def _suggest_missing_engine_boxes(
    *,
    args: argparse.Namespace,
    class_map: Dict[int, str],
    cls_ids: List[int],
    boxes_xyxy: List[List[float]],
    keep_mask: List[bool],
    image_stem: str,
    img_w: int,
    img_h: int,
) -> List[Tuple[int, List[float]]]:
    if not bool(args.suggest_missing_engines):
        return []
    if not _name_matches_stem_filter(str(image_stem), str(args.suggest_engine_stem_contains)):
        return []
    if img_w <= 0 or img_h <= 0:
        return []

    aircraft_cls_id = _find_aircraft_class_id(class_map)
    left_cls_id, right_cls_id = _find_engine_class_ids(class_map)
    if left_cls_id is None or right_cls_id is None:
        return []

    left_idx = _find_largest_kept_box_index_for_class(
        cls_ids=cls_ids,
        boxes_xyxy=boxes_xyxy,
        keep_mask=keep_mask,
        target_cls_id=left_cls_id,
    )
    right_idx = _find_largest_kept_box_index_for_class(
        cls_ids=cls_ids,
        boxes_xyxy=boxes_xyxy,
        keep_mask=keep_mask,
        target_cls_id=right_cls_id,
    )
    if (left_idx is None and right_idx is None) or (left_idx is not None and right_idx is not None):
        return []

    aircraft_idx = _find_largest_kept_box_index_for_class(
        cls_ids=cls_ids,
        boxes_xyxy=boxes_xyxy,
        keep_mask=keep_mask,
        target_cls_id=aircraft_cls_id,
    )
    if aircraft_idx is None:
        return []

    source_idx: int = int(left_idx if left_idx is not None else right_idx)
    missing_cls_id: int = int(right_cls_id if left_idx is not None else left_cls_id)

    source = [float(v) for v in boxes_xyxy[source_idx]]
    aircraft = [float(v) for v in boxes_xyxy[aircraft_idx]]
    mirror_axis_x = 0.5 * (aircraft[0] + aircraft[2])

    mirrored = [
        2.0 * mirror_axis_x - source[2],
        source[1],
        2.0 * mirror_axis_x - source[0],
        source[3],
    ]
    norm = _normalize_bbox_xyxy(mirrored, img_w=img_w, img_h=img_h, min_size_px=2.0)
    if norm is None:
        return []
    return [(missing_cls_id, norm)]


def _suggest_missing_front_gear_box(
    *,
    args: argparse.Namespace,
    class_map: Dict[int, str],
    cls_ids: List[int],
    boxes_xyxy: List[List[float]],
    keep_mask: List[bool],
    image_stem: str,
    img_w: int,
    img_h: int,
) -> Optional[Tuple[int, List[float]]]:
    if not bool(args.suggest_missing_front_gear):
        return None
    if not _name_matches_stem_filter(str(image_stem), str(args.suggest_front_gear_stem_contains)):
        return None
    if img_w <= 0 or img_h <= 0:
        return None

    front_cls_id = _find_front_gear_class_id(class_map)
    if front_cls_id is None:
        return None

    for i, cls_id in enumerate(cls_ids):
        if not bool(keep_mask[i]):
            continue
        if int(cls_id) == int(front_cls_id):
            return None

    left_idx: Optional[int] = None
    right_idx: Optional[int] = None
    left_area = -1.0
    right_area = -1.0

    for i, cls_id in enumerate(cls_ids):
        if not bool(keep_mask[i]):
            continue
        side = _engine_side_for_class(int(cls_id), class_map)
        if side is None:
            continue
        box = boxes_xyxy[i]
        area = max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))
        if side == "left" and area > left_area:
            left_area = area
            left_idx = i
        elif side == "right" and area > right_area:
            right_area = area
            right_idx = i

    if left_idx is None or right_idx is None:
        return None

    lbox = [float(v) for v in boxes_xyxy[left_idx]]
    rbox = [float(v) for v in boxes_xyxy[right_idx]]
    lc_x = 0.5 * (lbox[0] + lbox[2])
    lc_y = 0.5 * (lbox[1] + lbox[3])
    rc_x = 0.5 * (rbox[0] + rbox[2])
    rc_y = 0.5 * (rbox[1] + rbox[3])
    lw = max(2.0, lbox[2] - lbox[0])
    rw = max(2.0, rbox[2] - rbox[0])
    lh = max(2.0, lbox[3] - lbox[1])
    rh = max(2.0, rbox[3] - rbox[1])

    mid_x = 0.5 * (lc_x + rc_x)
    mid_y = 0.5 * (lc_y + rc_y)
    span_x = max(2.0, abs(rc_x - lc_x))
    engine_w = max(2.0, 0.5 * (lw + rw))
    engine_h = max(2.0, 0.5 * (lh + rh))

    pred_x = mid_x + float(args.suggest_front_gear_dx_over_engine_span) * span_x
    pred_y = mid_y + float(args.suggest_front_gear_dy_over_engine_h) * engine_h
    pred_w = max(2.0, float(args.suggest_front_gear_w_over_engine_w) * engine_w)
    pred_h = max(2.0, float(args.suggest_front_gear_h_over_engine_h) * engine_h)
    pred_xyxy = [
        pred_x - 0.5 * pred_w,
        pred_y - 0.5 * pred_h,
        pred_x + 0.5 * pred_w,
        pred_y + 0.5 * pred_h,
    ]
    norm = _normalize_bbox_xyxy(pred_xyxy, img_w=img_w, img_h=img_h, min_size_px=2.0)
    if norm is None:
        return None
    return int(front_cls_id), norm


def _load_class_names(yaml_path: Path) -> Dict[int, str]:
    if not yaml_path.exists() or not yaml_path.is_file():
        return {0: "aircraft", 1: "engine_left", 2: "engine_right"}

    lines = yaml_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    class_map: Dict[int, str] = {}
    in_names = False
    names_indent = 0

    for ln in lines:
        raw = ln.rstrip("\n")
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if stripped.startswith("names:"):
            rhs = stripped.split(":", 1)[1].strip()
            if rhs:
                # inline list or dict
                try:
                    obj = ast.literal_eval(rhs)
                    if isinstance(obj, list):
                        class_map = {i: str(v) for i, v in enumerate(obj)}
                        return class_map
                    if isinstance(obj, dict):
                        out: Dict[int, str] = {}
                        for k, v in obj.items():
                            out[int(k)] = str(v)
                        return out
                except Exception:
                    pass
            in_names = True
            names_indent = len(raw) - len(raw.lstrip(" "))
            continue

        if in_names:
            indent = len(raw) - len(raw.lstrip(" "))
            if indent <= names_indent:
                break
            entry = stripped
            if ":" in entry:
                k, v = entry.split(":", 1)
                k = k.strip()
                v = v.strip().strip("\"'")
                if k.isdigit():
                    class_map[int(k)] = v
            elif entry.startswith("- "):
                # block list style under names
                class_map[len(class_map)] = entry[2:].strip().strip("\"'")

    if class_map:
        return class_map
    return {0: "aircraft", 1: "engine_left", 2: "engine_right"}


def _class_color_name(cls_id: int, class_map: Dict[int, str]) -> Tuple[Tuple[int, int, int], str]:
    name = str(class_map.get(int(cls_id), f"class_{int(cls_id)}"))
    lname = _normalize_class_token(name)
    if lname == "aircraft":
        return (0, 255, 0), name
    if lname in {"engine_left", "left_engine"}:
        return (255, 200, 0), name
    if lname in {"engine_right", "right_engine"}:
        return (0, 200, 255), name
    if lname in {
        "front_gear",
        "frontgear",
        "frontwheel",
        "front_wheel",
        "front_wheels",
        "front_landing_gear",
        "nose_gear",
        "nose_wheel",
    }:
        return (0, 80, 255), name
    return (255, 255, 255), name


def _effective_params(args: argparse.Namespace, cls_id: int, class_map: Dict[int, str]) -> Tuple[float, float, float]:
    name = _normalize_class_token(str(class_map.get(int(cls_id), "")))
    scale = float(args.scale_all)
    sx = float(args.shift_x_all)
    sy = float(args.shift_y_all)

    if name == "aircraft" or int(cls_id) == 0:
        scale *= float(args.scale_aircraft)
        sx += float(args.shift_x_aircraft)
        sy += float(args.shift_y_aircraft)
    elif name in {"engine_left", "left_engine"} or int(cls_id) == 1:
        scale *= float(args.scale_engine_left)
        sx += float(args.shift_x_engine_left)
        sy += float(args.shift_y_engine_left)
    elif name in {"engine_right", "right_engine"} or int(cls_id) == 2:
        scale *= float(args.scale_engine_right)
        sx += float(args.shift_x_engine_right)
        sy += float(args.shift_y_engine_right)

    return scale, sx, sy


def _name_matches_stem_filter(stem: str, stem_filter: str) -> bool:
    filt = str(stem_filter or "").strip().lower()
    if not filt:
        return True
    return filt in str(stem or "").lower()


def _is_engine_class(cls_id: int, class_map: Dict[int, str]) -> bool:
    name = _normalize_class_token(str(class_map.get(int(cls_id), "")))
    if int(cls_id) in {1, 2}:
        return True
    return name in {"engine_left", "left_engine", "engine_right", "right_engine"}


def _override_engine_size_if_requested(
    *,
    args: argparse.Namespace,
    class_map: Dict[int, str],
    cls_id: int,
    image_stem: str,
    img_w: int,
    img_h: int,
    bw: float,
    bh: float,
) -> Tuple[float, float, bool]:
    target_w_px = float(args.engine_fixed_w_px)
    target_h_px = float(args.engine_fixed_h_px)
    if target_w_px <= 0.0 or target_h_px <= 0.0:
        return float(bw), float(bh), False
    if img_w <= 0 or img_h <= 0:
        return float(bw), float(bh), False
    if not _name_matches_stem_filter(image_stem, str(args.stem_contains)):
        return float(bw), float(bh), False
    if not _is_engine_class(int(cls_id), class_map):
        return float(bw), float(bh), False

    bw_fixed = float(target_w_px) / float(img_w)
    bh_fixed = float(target_h_px) / float(img_h)
    return float(bw_fixed), float(bh_fixed), True


def _ensure_dirs(out_root: Path, make_viz: bool) -> None:
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


def _write_yaml(out_yaml: Path, out_root: Path, class_map: Dict[int, str]) -> None:
    lines = [
        "# YOLO detection dataset — edited bboxes",
        f"path: {out_root}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        "names:",
    ]
    for k in sorted(class_map.keys()):
        lines.append(f"  {int(k)}: {class_map[k]}")
    out_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _selected_splits(split_arg: str) -> List[str]:
    if str(split_arg).lower() == "all":
        return ["train", "val", "test"]
    return [str(split_arg).lower()]


def _collect_image_records(
    dataset_root: Path,
    out_root: Path,
    split_arg: str,
    start_index: int,
    max_images: int,
) -> List[Tuple[str, Path, Path, Path, Path, Path]]:
    records: List[Tuple[str, Path, Path, Path, Path, Path]] = []
    for split in _selected_splits(split_arg):
        src_img_dir = dataset_root / "images" / split
        src_lbl_dir = dataset_root / "labels" / split
        dst_img_dir = out_root / "images" / split
        dst_lbl_dir = out_root / "labels" / split
        dst_vis_dir = out_root / "vis" / split
        if not src_img_dir.exists() or not src_img_dir.is_dir():
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
            records.append(
                (
                    split,
                    img_path,
                    src_lbl_dir / f"{stem}.txt",
                    dst_img_dir / img_path.name,
                    dst_lbl_dir / f"{stem}.txt",
                    dst_vis_dir / img_path.name,
                )
            )

    start = max(0, int(start_index))
    if start > 0:
        records = records[start:]
    if int(max_images) > 0:
        records = records[: int(max_images)]
    return records


def _load_boxes_for_image(
    args: argparse.Namespace,
    class_map: Dict[int, str],
    label_path: Path,
    img_w: int,
    img_h: int,
    image_stem: str,
) -> Tuple[List[int], List[List[float]], List[bool], int]:
    cls_ids: List[int] = []
    boxes_xyxy: List[List[float]] = []
    keep_mask: List[bool] = []
    skipped = 0

    raw_lines: List[str] = []
    if label_path.exists() and label_path.is_file():
        raw_lines = label_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    for ln in raw_lines:
        parsed = _parse_det_line(ln)
        if parsed is None:
            skipped += 1
            continue
        cls_id, xc, yc, bw, bh = parsed

        # Keep same pre-transform behavior as non-interactive path.
        scale, sx, sy = _effective_params(args, cls_id=cls_id, class_map=class_map)
        if scale <= 0.0:
            skipped += 1
            continue
        bw2 = float(bw) * float(scale)
        bh2 = float(bh) * float(scale)
        xc2 = float(xc) + float(sx)
        yc2 = float(yc) + float(sy)
        bw2, bh2, _ = _override_engine_size_if_requested(
            args=args,
            class_map=class_map,
            cls_id=int(cls_id),
            image_stem=str(image_stem),
            img_w=int(img_w),
            img_h=int(img_h),
            bw=float(bw2),
            bh=float(bh2),
        )
        clipped = _clip_xywhn(xc2, yc2, bw2, bh2)
        if clipped is None:
            skipped += 1
            continue
        xyxy = _xywhn_to_xyxy_float(*clipped, img_w=img_w, img_h=img_h)
        if xyxy is None:
            skipped += 1
            continue
        norm = _normalize_bbox_xyxy(xyxy, img_w=img_w, img_h=img_h, min_size_px=1.0)
        if norm is None:
            skipped += 1
            continue
        cls_ids.append(int(cls_id))
        boxes_xyxy.append(norm)
        keep_mask.append(True)
    return cls_ids, boxes_xyxy, keep_mask, skipped


def _run_interactive_mode(
    args: argparse.Namespace,
    dataset_root: Path,
    out_root: Path,
    class_map: Dict[int, str],
    make_viz: bool,
) -> Tuple[int, int, int, int, int, int, int, int]:
    records = _collect_image_records(
        dataset_root=dataset_root,
        out_root=out_root,
        split_arg=str(args.split),
        start_index=int(args.start_index),
        max_images=int(args.max_images),
    )
    if not records:
        raise RuntimeError("No images found for interactive mode with current split/start/max filters.")

    window_name = "bbox_editor"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    total_images = 0
    total_labels = 0
    total_boxes_in = 0
    total_boxes_out = 0
    dropped_boxes = 0
    missing_labels = 0
    bad_images = 0
    viz_saved = 0

    quit_all = False
    for rec_idx, (split, src_img_path, src_lbl_path, dst_img_path, dst_lbl_path, dst_vis_path) in enumerate(records):
        img = cv2.imread(str(src_img_path))
        if img is None:
            bad_images += 1
            print(f"[interactive][skip] unreadable image: {src_img_path}")
            continue

        h, w = img.shape[:2]
        if not src_lbl_path.exists():
            missing_labels += 1

        cls_ids, boxes_xyxy, keep_mask, skipped = _load_boxes_for_image(
            args=args,
            class_map=class_map,
            label_path=src_lbl_path,
            img_w=w,
            img_h=h,
            image_stem=src_img_path.stem,
        )
        total_boxes_in += len(cls_ids) + int(skipped)
        suggested_mask: List[bool] = [False] * len(cls_ids)
        engine_suggestions = _suggest_missing_engine_boxes(
            args=args,
            class_map=class_map,
            cls_ids=cls_ids,
            boxes_xyxy=boxes_xyxy,
            keep_mask=keep_mask,
            image_stem=src_img_path.stem,
            img_w=w,
            img_h=h,
        )
        front_suggestion = _suggest_missing_front_gear_box(
            args=args,
            class_map=class_map,
            cls_ids=cls_ids,
            boxes_xyxy=boxes_xyxy,
            keep_mask=keep_mask,
            image_stem=src_img_path.stem,
            img_w=w,
            img_h=h,
        )
        for sug_cls, sug_box in engine_suggestions:
            cls_ids.append(int(sug_cls))
            boxes_xyxy.append(list(sug_box))
            keep_mask.append(bool(args.suggest_engine_keep))
            suggested_mask.append(True)
        if front_suggestion is not None:
            sug_cls, sug_box = front_suggestion
            cls_ids.append(int(sug_cls))
            boxes_xyxy.append(list(sug_box))
            keep_mask.append(bool(args.suggest_front_gear_keep))
            suggested_mask.append(True)
        print(
            f"[interactive] opened {rec_idx + 1}/{len(records)} split={split} "
            f"image={src_img_path} label={src_lbl_path} "
            f"boxes={len(cls_ids)} skipped={int(skipped)} suggested={sum(1 for s in suggested_mask if s)}"
        )
        selected_idx: Optional[int] = 0 if boxes_xyxy else None
        drag_handle: Optional[str] = None
        drag_start_box: Optional[List[float]] = None
        drag_anchor: Tuple[float, float] = (0.0, 0.0)

        # For reset key.
        orig_cls_ids = list(cls_ids)
        orig_boxes = [list(b) for b in boxes_xyxy]
        orig_keep = list(keep_mask)
        orig_suggested = list(suggested_mask)

        title = (
            f"{split} {rec_idx + 1}/{len(records)} {src_img_path.name} "
            f"boxes={sum(1 for k in keep_mask if k)}"
        )

        def _mouse_cb(event: int, x: int, y: int, flags: int, param: object) -> None:
            nonlocal selected_idx, drag_handle, drag_start_box, drag_anchor, boxes_xyxy, keep_mask
            px, py = float(x), float(y)

            if event == cv2.EVENT_LBUTTONDOWN:
                best_idx: Optional[int] = None
                best_handle: Optional[str] = None
                # Prefer top-most (last) active box.
                for i in range(len(boxes_xyxy) - 1, -1, -1):
                    if not keep_mask[i]:
                        continue
                    handle = _pick_bbox_handle(boxes_xyxy[i], px, py, radius=10.0)
                    if handle is not None:
                        best_idx = i
                        best_handle = handle
                        break
                if best_idx is not None and best_handle is not None:
                    selected_idx = best_idx
                    drag_handle = best_handle
                    drag_anchor = (px, py)
                    drag_start_box = list(boxes_xyxy[best_idx])

            elif event == cv2.EVENT_MOUSEMOVE:
                if drag_handle is not None and drag_start_box is not None and selected_idx is not None:
                    new_box = _apply_bbox_drag(drag_start_box, drag_handle, drag_anchor, (px, py))
                    norm = _normalize_bbox_xyxy(new_box, img_w=w, img_h=h, min_size_px=2.0)
                    if norm is not None:
                        boxes_xyxy[selected_idx] = norm

            elif event == cv2.EVENT_LBUTTONUP:
                drag_handle = None
                drag_start_box = None

            elif event == cv2.EVENT_RBUTTONDOWN:
                for i in range(len(boxes_xyxy) - 1, -1, -1):
                    handle = _pick_bbox_handle(boxes_xyxy[i], px, py, radius=10.0)
                    if handle is not None:
                        selected_idx = i
                        keep_mask[i] = not bool(keep_mask[i])
                        break

        cv2.setMouseCallback(window_name, _mouse_cb)

        saved_once = False
        while True:
            title = (
                f"{split} {rec_idx + 1}/{len(records)} {src_img_path.name} "
                f"kept={sum(1 for k in keep_mask if k)}/{len(keep_mask)}"
            )
            canvas = _draw_boxes_canvas(
                image=img,
                boxes_xyxy=boxes_xyxy,
                cls_ids=cls_ids,
                keep_mask=keep_mask,
                class_map=class_map,
                suggested_mask=suggested_mask,
                selected_idx=selected_idx,
                show_handles=True,
                title=title,
            )
            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(20) & 0xFF
            if key == 255:
                continue
            if key in (9, ord("k")):  # tab / k
                if boxes_xyxy:
                    if selected_idx is None:
                        selected_idx = 0
                    else:
                        selected_idx = (int(selected_idx) + 1) % len(boxes_xyxy)
                continue
            if key == ord("x"):
                if selected_idx is not None and 0 <= int(selected_idx) < len(keep_mask):
                    keep_mask[selected_idx] = not bool(keep_mask[selected_idx])
                continue
            if key == ord("r"):
                cls_ids = list(orig_cls_ids)
                boxes_xyxy = [list(b) for b in orig_boxes]
                keep_mask = list(orig_keep)
                suggested_mask = list(orig_suggested)
                selected_idx = 0 if boxes_xyxy else None
                continue

            def _save_current(action_key: str) -> None:
                nonlocal total_images, total_labels, total_boxes_out, dropped_boxes, viz_saved, saved_once
                dst_img_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_img_path, dst_img_path)
                kept = _write_det_labels_from_xyxy(
                    out_label_path=dst_lbl_path,
                    boxes_xyxy=boxes_xyxy,
                    cls_ids=cls_ids,
                    keep_mask=keep_mask,
                    img_w=w,
                    img_h=h,
                )
                if make_viz:
                    dst_vis_path.parent.mkdir(parents=True, exist_ok=True)
                    vis_img = _draw_boxes_canvas(
                        image=img,
                        boxes_xyxy=boxes_xyxy,
                        cls_ids=cls_ids,
                        keep_mask=keep_mask,
                        class_map=class_map,
                        suggested_mask=suggested_mask,
                        selected_idx=None,
                        show_handles=False,
                        title="",
                    )
                    cv2.imwrite(str(dst_vis_path), vis_img)
                    viz_saved += 1
                if not saved_once:
                    num_suggested = sum(1 for s in suggested_mask if s)
                    num_suggested_kept = sum(
                        1 for i, s in enumerate(suggested_mask) if s and i < len(keep_mask) and bool(keep_mask[i])
                    )
                    total_images += 1
                    total_labels += 1
                    total_boxes_out += int(kept)
                    original_boxes = max(0, len(cls_ids) - int(num_suggested))
                    original_kept = max(0, int(kept) - int(num_suggested_kept))
                    dropped_boxes += max(0, int(original_boxes) - int(original_kept))
                    saved_once = True
                print(
                    f"[interactive] saved ({action_key}): {dst_lbl_path} "
                    f"kept={int(kept)}/{len(cls_ids)} vis={bool(make_viz)}"
                )

            if key == ord("s"):
                _save_current("s")
                continue
            if key == ord("n"):
                _save_current("n")
                break
            if key == ord("q"):
                _save_current("q")
                quit_all = True
                break

        if quit_all:
            break

    cv2.destroyWindow(window_name)
    return (
        total_images,
        total_labels,
        total_boxes_in,
        total_boxes_out,
        dropped_boxes,
        missing_labels,
        bad_images,
        viz_saved,
    )


def main() -> None:
    args = parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    out_root = Path(args.out_dir).expanduser().resolve()
    if dataset_root == out_root:
        raise RuntimeError("--out-dir must be different from --dataset-root")

    in_yaml = dataset_root / str(args.yaml_name).strip()
    class_map = _load_class_names(in_yaml)

    make_viz = bool(args.make_viz)
    _ensure_dirs(out_root, make_viz=make_viz)

    total_images = 0
    total_labels = 0
    total_boxes_in = 0
    total_boxes_out = 0
    dropped_boxes = 0
    missing_labels = 0
    bad_images = 0
    viz_saved = 0

    if bool(args.interactive):
        (
            total_images,
            total_labels,
            total_boxes_in,
            total_boxes_out,
            dropped_boxes,
            missing_labels,
            bad_images,
            viz_saved,
        ) = _run_interactive_mode(
            args=args,
            dataset_root=dataset_root,
            out_root=out_root,
            class_map=class_map,
            make_viz=make_viz,
        )
    else:
        records = _collect_image_records(
            dataset_root=dataset_root,
            out_root=out_root,
            split_arg=str(args.split),
            start_index=int(args.start_index),
            max_images=int(args.max_images),
        )

        for split, img_path, src_lbl, dst_img_path, dst_lbl_path, dst_vis_path in records:
            raw_lines: List[str] = []
            if src_lbl.exists() and src_lbl.is_file():
                raw_lines = src_lbl.read_text(encoding="utf-8", errors="ignore").splitlines()
            else:
                missing_labels += 1

            edited_lines: List[str] = []
            edited_boxes_xyxy: List[List[float]] = []
            edited_cls_ids: List[int] = []
            edited_keep: List[bool] = []

            img = cv2.imread(str(img_path))
            if img is None:
                bad_images += 1
                print(f"[skip] unreadable image: {img_path}")
                continue
            h, w = img.shape[:2]

            for ln in raw_lines:
                parsed = _parse_det_line(ln)
                if parsed is None:
                    continue
                cls_id, xc, yc, bw, bh = parsed
                total_boxes_in += 1

                scale, sx, sy = _effective_params(args, cls_id=cls_id, class_map=class_map)
                if scale <= 0.0:
                    dropped_boxes += 1
                    continue

                new_bw = float(bw) * float(scale)
                new_bh = float(bh) * float(scale)
                new_xc = float(xc) + float(sx)
                new_yc = float(yc) + float(sy)
                new_bw, new_bh, _ = _override_engine_size_if_requested(
                    args=args,
                    class_map=class_map,
                    cls_id=int(cls_id),
                    image_stem=img_path.stem,
                    img_w=int(w),
                    img_h=int(h),
                    bw=float(new_bw),
                    bh=float(new_bh),
                )
                clipped = _clip_xywhn(new_xc, new_yc, new_bw, new_bh)
                if clipped is None:
                    dropped_boxes += 1
                    continue

                edited_lines.append(_fmt_line(cls_id, *clipped))
                total_boxes_out += 1

                xyxy = _xywhn_to_xyxy_float(*clipped, img_w=w, img_h=h)
                if xyxy is not None:
                    norm = _normalize_bbox_xyxy(xyxy, img_w=w, img_h=h, min_size_px=1.0)
                    if norm is not None:
                        edited_boxes_xyxy.append(norm)
                        edited_cls_ids.append(int(cls_id))
                        edited_keep.append(True)

            dst_img_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, dst_img_path)
            total_images += 1

            dst_lbl_path.parent.mkdir(parents=True, exist_ok=True)
            dst_lbl_path.write_text(
                ("\n".join(edited_lines) + "\n") if edited_lines else "",
                encoding="utf-8",
            )
            total_labels += 1

            if make_viz:
                dst_vis_path.parent.mkdir(parents=True, exist_ok=True)
                vis_img = _draw_boxes_canvas(
                    image=img,
                    boxes_xyxy=edited_boxes_xyxy,
                    cls_ids=edited_cls_ids,
                    keep_mask=edited_keep,
                    class_map=class_map,
                    selected_idx=None,
                    show_handles=False,
                    title="",
                )
                cv2.imwrite(str(dst_vis_path), vis_img)
                viz_saved += 1

    out_yaml = out_root / str(args.yaml_name).strip()
    _write_yaml(out_yaml=out_yaml, out_root=out_root, class_map=class_map)

    print(f"[done] output: {out_root}")
    print(f"[done] yaml:   {out_yaml}")
    if make_viz:
        print(f"[done] vis:    {out_root / 'vis'}")
    print(
        f"[summary] images={total_images} labels={total_labels} "
        f"boxes_in={total_boxes_in} boxes_out={total_boxes_out} dropped_boxes={dropped_boxes}"
    )
    if make_viz:
        print(f"[summary] vis_images={viz_saved} bad_images_for_vis={bad_images}")
    print(f"[summary] missing_label_files={missing_labels}")


if __name__ == "__main__":
    main()
