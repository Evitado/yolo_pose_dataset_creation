#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interactive pixel-level editor for YOLO pose labels.

Features:
- Open one image + matching label.
- Or pass a folder and edit images one-by-one.
- Drag keypoints to correct pixel locations.
- Toggle keypoint visibility.
- Move/resize aircraft bbox.
- Save back to YOLO pose label format.

Examples:
  python3 adjust_keypoint_pixels.py \
    --image ./aircraft_pose_with_normalising_applied_grayscale/images/train/scene.png

  python3 adjust_keypoint_pixels.py \
    --image ./aircraft_pose_with_normalising_applied_grayscale/images/train/scene.png \
    --label ./aircraft_pose_with_normalising_applied_grayscale/labels/train/scene.txt \
    --yaml ./aircraft_pose_with_normalising_applied_grayscale/aircraft_pose.yaml

  python3 adjust_keypoint_pixels.py \
    --image-dir ./aircraft_pose_with_normalising_applied_grayscale/images/train

  # Uses DEFAULT_IMAGE_DIR / DEFAULT_YAML_PATH from this file:
  python3 adjust_keypoint_pixels.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Default paths for quick use (you can edit these).
DEFAULT_DATASET_ROOT = Path("/home/femi/yolo_pose_dataset_creation/aircraft_pose_with_normalising_applied_multifield_only_3_2")
DEFAULT_IMAGE_DIR = DEFAULT_DATASET_ROOT / "images" / "val"
DEFAULT_YAML_PATH = DEFAULT_DATASET_ROOT / "aircraft_pose.yaml"


def _clamp(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


def _collect_images(image_dir: Path, recursive: bool) -> List[Path]:
    if not image_dir.exists() or not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    it = image_dir.rglob("*") if recursive else image_dir.glob("*")
    imgs = sorted([p for p in it if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
    if not imgs:
        raise RuntimeError(f"No images found under: {image_dir}")
    return imgs


def _infer_label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    if "images" not in parts:
        raise ValueError("Could not infer label path: image path must contain '/images/'.")
    i = parts.index("images")
    parts[i] = "labels"
    out = Path(*parts).with_suffix(".txt")
    return out


def _infer_yaml_path(image_path: Path) -> Optional[Path]:
    cur = image_path.resolve()
    for parent in [cur.parent] + list(cur.parents):
        cand = parent / "aircraft_pose.yaml"
        if cand.exists():
            return cand
    return None


def _load_keypoint_names_from_yaml(yaml_path: Optional[Path]) -> List[str]:
    if yaml_path is None or not yaml_path.exists():
        return []

    lines = yaml_path.read_text(encoding="utf-8", errors="ignore").splitlines()
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


def _bbox_xywhn_to_xyxy(cx: float, cy: float, bw: float, bh: float, w: int, h: int) -> np.ndarray:
    bw_px = float(bw) * float(w)
    bh_px = float(bh) * float(h)
    x1 = float(cx) * float(w) - 0.5 * bw_px
    y1 = float(cy) * float(h) - 0.5 * bh_px
    x2 = float(cx) * float(w) + 0.5 * bw_px
    y2 = float(cy) * float(h) + 0.5 * bh_px
    return np.array([x1, y1, x2, y2], dtype=np.float64)


def _bbox_xyxy_to_xywhn(bbox: np.ndarray, w: int, h: int) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    bw_px = max(1e-6, x2 - x1)
    bh_px = max(1e-6, y2 - y1)
    cx = (x1 + x2) * 0.5 / float(w)
    cy = (y1 + y2) * 0.5 / float(h)
    bw = bw_px / float(w)
    bh = bh_px / float(h)
    return (
        _clamp(cx, 0.0, 1.0),
        _clamp(cy, 0.0, 1.0),
        _clamp(bw, 1e-6, 1.0),
        _clamp(bh, 1e-6, 1.0),
    )


def _normalize_bbox_xyxy(bbox: np.ndarray, w: int, h: int) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    x1 = _clamp(x1, 0.0, float(w - 1))
    x2 = _clamp(x2, 0.0, float(w - 1))
    y1 = _clamp(y1, 0.0, float(h - 1))
    y2 = _clamp(y2, 0.0, float(h - 1))
    if x2 <= x1:
        x2 = min(float(w - 1), x1 + 1.0)
    if y2 <= y1:
        y2 = min(float(h - 1), y1 + 1.0)
    return np.array([x1, y1, x2, y2], dtype=np.float64)


def _parse_pose_line(line: str, w: int, h: int) -> Tuple[int, np.ndarray, np.ndarray]:
    toks = line.strip().split()
    if len(toks) < 5:
        raise ValueError("Invalid label line: expected at least class + bbox.")

    cls_id = int(float(toks[0]))
    cx, cy, bw, bh = [float(x) for x in toks[1:5]]
    bbox = _normalize_bbox_xyxy(_bbox_xywhn_to_xyxy(cx, cy, bw, bh, w, h), w, h)

    rem = toks[5:]
    if len(rem) % 3 != 0:
        raise ValueError("Invalid keypoint payload: expected triplets x y v.")

    n = len(rem) // 3
    kps = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        xn = float(rem[3 * i + 0])
        yn = float(rem[3 * i + 1])
        vv = float(rem[3 * i + 2])
        px = _clamp(xn * float(w), 0.0, float(w - 1))
        py = _clamp(yn * float(h), 0.0, float(h - 1))
        vis = 1.0 if vv > 0 else 0.0
        kps[i] = [px, py, vis]

    return cls_id, bbox, kps


def _serialize_pose_line(cls_id: int, bbox: np.ndarray, kps: np.ndarray, w: int, h: int) -> str:
    cx, cy, bw, bh = _bbox_xyxy_to_xywhn(bbox, w, h)
    parts: List[str] = [str(int(cls_id)), f"{cx:.6f}", f"{cy:.6f}", f"{bw:.6f}", f"{bh:.6f}"]
    for i in range(kps.shape[0]):
        x = _clamp(float(kps[i, 0]), 0.0, float(w - 1))
        y = _clamp(float(kps[i, 1]), 0.0, float(h - 1))
        v = 1 if float(kps[i, 2]) > 0.5 else 0
        xn = _clamp(x / float(w), 0.0, 1.0)
        yn = _clamp(y / float(h), 0.0, 1.0)
        if v == 0:
            parts.extend(["0.000000", "0.000000", "0"])
        else:
            parts.extend([f"{xn:.6f}", f"{yn:.6f}", "1"])
    return " ".join(parts)


def _choose_label_line(lines: List[str], line_index: int) -> int:
    nonempty = [i for i, ln in enumerate(lines) if ln.strip()]
    if not nonempty:
        raise ValueError("Label file has no non-empty lines.")
    if line_index < 0 or line_index >= len(nonempty):
        raise ValueError(f"--line-index out of range: {line_index}, available={len(nonempty)}")
    return nonempty[line_index]


def _pick_nearest_keypoint(kps: np.ndarray, x: float, y: float, max_dist: float) -> Optional[int]:
    if kps.size == 0:
        return None
    d2 = (kps[:, 0] - x) ** 2 + (kps[:, 1] - y) ** 2
    i = int(np.argmin(d2))
    if float(np.sqrt(d2[i])) <= float(max_dist):
        return i
    return None


def _bbox_corners(bbox: np.ndarray) -> dict:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return {
        "tl": (x1, y1),
        "tr": (x2, y1),
        "br": (x2, y2),
        "bl": (x1, y2),
    }


def _pick_bbox_handle(bbox: np.ndarray, x: float, y: float, radius: float) -> Optional[str]:
    corners = _bbox_corners(bbox)
    for name, (cx, cy) in corners.items():
        if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
            return name
    x1, y1, x2, y2 = [float(v) for v in bbox]
    if x1 <= x <= x2 and y1 <= y <= y2:
        return "move"
    return None


def _apply_bbox_drag(start_bbox: np.ndarray, handle: str, anchor: Tuple[float, float], cur: Tuple[float, float]) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in start_bbox]
    ax, ay = anchor
    cx, cy = cur
    dx, dy = cx - ax, cy - ay

    if handle == "move":
        return np.array([x1 + dx, y1 + dy, x2 + dx, y2 + dy], dtype=np.float64)
    if handle == "tl":
        return np.array([cx, cy, x2, y2], dtype=np.float64)
    if handle == "tr":
        return np.array([x1, cy, cx, y2], dtype=np.float64)
    if handle == "br":
        return np.array([x1, y1, cx, cy], dtype=np.float64)
    if handle == "bl":
        return np.array([cx, y1, x2, cy], dtype=np.float64)
    return start_bbox.copy()


@dataclass
class EditorState:
    image: np.ndarray
    label_lines: List[str]
    label_line_idx: int
    class_id: int
    bbox_xyxy: np.ndarray
    kps_xyv: np.ndarray
    kp_names: List[str]
    pick_radius: float
    window_name: str
    out_label: Path
    selected_idx: int = 0
    mode: str = "keypoint"  # keypoint | bbox
    dirty: bool = False
    drag_kp: bool = False
    drag_bbox_handle: Optional[str] = None
    drag_anchor: Tuple[float, float] = (0.0, 0.0)
    drag_bbox_start: Optional[np.ndarray] = None
    mouse_xy: Tuple[int, int] = (0, 0)
    orig_bbox_xyxy: Optional[np.ndarray] = None
    orig_kps_xyv: Optional[np.ndarray] = None
    show_hud: bool = True
    hud_height: int = 64

    @property
    def h(self) -> int:
        return int(self.image.shape[0])

    @property
    def w(self) -> int:
        return int(self.image.shape[1])


def _draw(state: EditorState) -> np.ndarray:
    img_canvas = state.image.copy()
    x1, y1, x2, y2 = [int(round(v)) for v in state.bbox_xyxy]
    cv2.rectangle(img_canvas, (x1, y1), (x2, y2), (0, 220, 0), 2, lineType=cv2.LINE_AA)

    corners = _bbox_corners(state.bbox_xyxy)
    for _, (cx, cy) in corners.items():
        cv2.circle(img_canvas, (int(round(cx)), int(round(cy))), 5, (255, 255, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(img_canvas, (int(round(cx)), int(round(cy))), 5, (0, 0, 0), 1, lineType=cv2.LINE_AA)

    # Spread hidden keypoints that overlap at same pixel.
    # If they are at (0,0), display them around image center for easier editing.
    draw_xy = state.kps_xyv[:, :2].copy()
    hidden_idx = [i for i in range(state.kps_xyv.shape[0]) if state.kps_xyv[i, 2] <= 0.5]
    if hidden_idx:
        buckets: dict[tuple[int, int], List[int]] = {}
        for i in hidden_idx:
            key = (int(round(draw_xy[i, 0])), int(round(draw_xy[i, 1])))
            buckets.setdefault(key, []).append(i)
        for key, ids in buckets.items():
            if len(ids) <= 1:
                # Draw lone hidden (0,0) marker in center for visibility.
                if key == (0, 0):
                    i = ids[0]
                    draw_xy[i, 0] = float(state.w) * 0.5
                    draw_xy[i, 1] = float(state.h) * 0.5
                continue
            if key == (0, 0):
                base_x = float(state.w) * 0.5
                base_y = float(state.h) * 0.5
            else:
                base_x = float(key[0])
                base_y = float(key[1])
            for j, i in enumerate(ids):
                ox = ((j % 4) - 1.5) * 8.0
                oy = ((j // 4) + 1) * 8.0
                draw_xy[i, 0] = _clamp(base_x + ox, 0.0, float(state.w - 1))
                draw_xy[i, 1] = _clamp(base_y + oy, 0.0, float(state.h - 1))

    for i in range(state.kps_xyv.shape[0]):
        x = int(round(float(draw_xy[i, 0])))
        y = int(round(float(draw_xy[i, 1])))
        vis = int(state.kps_xyv[i, 2] > 0.5)
        selected = (i == state.selected_idx)

        if vis:
            col = (0, 255, 255) if selected else (0, 0, 255)
            cv2.circle(img_canvas, (x, y), 5 if selected else 4, col, -1, lineType=cv2.LINE_AA)
            cv2.circle(img_canvas, (x, y), 6, (0, 0, 0), 1, lineType=cv2.LINE_AA)
        else:
            col = (200, 200, 200) if selected else (130, 130, 130)
            cv2.line(img_canvas, (x - 4, y - 4), (x + 4, y + 4), col, 2, lineType=cv2.LINE_AA)
            cv2.line(img_canvas, (x - 4, y + 4), (x + 4, y - 4), col, 2, lineType=cv2.LINE_AA)

        name = state.kp_names[i] if i < len(state.kp_names) else f"kp_{i}"
        cv2.putText(
            img_canvas,
            f"{i}:{name}",
            (x + 7, y - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    if state.show_hud:
        kp_entries: List[str] = []
        for i in range(state.kps_xyv.shape[0]):
            name = state.kp_names[i] if i < len(state.kp_names) else f"kp_{i}"
            vis = "V" if state.kps_xyv[i, 2] > 0.5 else "H"
            sel = "*" if i == state.selected_idx else " "
            kp_entries.append(f"{sel}{i}:{name}[{vis}]")
        kp_lines = [" | ".join(kp_entries[i : i + 3]) for i in range(0, len(kp_entries), 3)]

        state.hud_height = 64 + 18 * len(kp_lines)
        canvas = np.zeros((state.h + state.hud_height, state.w, 3), dtype=np.uint8)
        canvas[: state.hud_height] = (20, 20, 20)
        canvas[state.hud_height :] = img_canvas

        sel_name = state.kp_names[state.selected_idx] if state.selected_idx < len(state.kp_names) else f"kp_{state.selected_idx}"
        info1 = f"Mode={state.mode} | Selected={state.selected_idx}:{sel_name} | Dirty={int(state.dirty)}"
        info2 = "Keys: n/p next/prev, v toggle vis, k keypoint, b bbox, r reset, s save, i HUD, q quit"
        info3 = "Mouse: left drag move, right toggle visibility (keypoint mode), n/p cycles all keypoints"

        def _hud_text(text: str, org: Tuple[int, int], color: Tuple[int, int, int]) -> None:
            cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        _hud_text(info1, (8, 18), (0, 230, 255))
        _hud_text(info2, (8, 36), (230, 230, 230))
        _hud_text(info3, (8, 54), (230, 230, 230))
        y0 = 72
        for i, ln in enumerate(kp_lines):
            _hud_text(ln, (8, y0 + i * 18), (170, 230, 170))
        return canvas

    state.hud_height = 0
    return img_canvas


def _display_to_image_xy(state: EditorState, x: float, y: float) -> tuple[float, float, bool]:
    y_off = float(state.hud_height if state.show_hud else 0)
    px = float(x)
    py = float(y) - y_off
    inside = (0.0 <= px <= float(state.w - 1)) and (0.0 <= py <= float(state.h - 1))
    px = _clamp(px, 0.0, float(state.w - 1))
    py = _clamp(py, 0.0, float(state.h - 1))
    return px, py, inside


def _save(state: EditorState) -> None:
    line = _serialize_pose_line(
        cls_id=state.class_id,
        bbox=state.bbox_xyxy,
        kps=state.kps_xyv,
        w=state.w,
        h=state.h,
    )
    out_lines = list(state.label_lines)
    out_lines[state.label_line_idx] = line
    state.out_label.parent.mkdir(parents=True, exist_ok=True)
    state.out_label.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    state.dirty = False
    print(f"[saved] {state.out_label}")


def _build_mouse_cb(state: EditorState):
    def _cb(event, x, y, flags, param):
        _ = flags, param
        state.mouse_xy = (int(x), int(y))
        px, py, inside = _display_to_image_xy(state, float(x), float(y))

        if event == cv2.EVENT_LBUTTONDOWN:
            if not inside:
                return
            if state.mode == "keypoint":
                idx = _pick_nearest_keypoint(state.kps_xyv, px, py, state.pick_radius)
                if idx is None:
                    idx = state.selected_idx
                state.selected_idx = int(idx)
                state.kps_xyv[idx, 0] = _clamp(px, 0.0, float(state.w - 1))
                state.kps_xyv[idx, 1] = _clamp(py, 0.0, float(state.h - 1))
                state.kps_xyv[idx, 2] = 1.0
                state.drag_kp = True
                state.dirty = True
            else:
                handle = _pick_bbox_handle(state.bbox_xyxy, px, py, radius=10.0)
                if handle is not None:
                    state.drag_bbox_handle = handle
                    state.drag_anchor = (px, py)
                    state.drag_bbox_start = state.bbox_xyxy.copy()

        elif event == cv2.EVENT_MOUSEMOVE:
            if state.drag_kp and state.mode == "keypoint":
                i = state.selected_idx
                state.kps_xyv[i, 0] = _clamp(px, 0.0, float(state.w - 1))
                state.kps_xyv[i, 1] = _clamp(py, 0.0, float(state.h - 1))
                state.kps_xyv[i, 2] = 1.0
                state.dirty = True
            elif state.drag_bbox_handle is not None and state.drag_bbox_start is not None and state.mode == "bbox":
                new_bbox = _apply_bbox_drag(state.drag_bbox_start, state.drag_bbox_handle, state.drag_anchor, (px, py))
                state.bbox_xyxy = _normalize_bbox_xyxy(new_bbox, state.w, state.h)
                state.dirty = True

        elif event == cv2.EVENT_LBUTTONUP:
            state.drag_kp = False
            state.drag_bbox_handle = None
            state.drag_bbox_start = None

        elif event == cv2.EVENT_RBUTTONDOWN:
            if not inside:
                return
            if state.mode == "keypoint":
                idx = _pick_nearest_keypoint(state.kps_xyv, px, py, state.pick_radius)
                if idx is None:
                    idx = state.selected_idx
                state.selected_idx = int(idx)
                if state.kps_xyv[idx, 2] > 0.5:
                    state.kps_xyv[idx, 2] = 0.0
                else:
                    state.kps_xyv[idx, 0] = _clamp(px, 0.0, float(state.w - 1))
                    state.kps_xyv[idx, 1] = _clamp(py, 0.0, float(state.h - 1))
                    state.kps_xyv[idx, 2] = 1.0
                state.dirty = True

    return _cb


def run(
    image_path: Path,
    label_path: Optional[Path],
    yaml_path: Optional[Path],
    out_label: Optional[Path],
    line_index: int,
    pick_radius: float,
    window_name: str,
) -> str:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    label_path = label_path if label_path is not None else _infer_label_path(image_path)
    if not label_path.exists():
        raise FileNotFoundError(f"Label not found: {label_path}")

    if yaml_path is None:
        yaml_path = _infer_yaml_path(image_path)

    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    h, w = img_bgr.shape[:2]

    lines = label_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    line_idx = _choose_label_line(lines, line_index)
    cls_id, bbox, kps = _parse_pose_line(lines[line_idx], w=w, h=h)

    yaml_names = _load_keypoint_names_from_yaml(yaml_path)
    if yaml_names:
        kp_names = list(yaml_names)
        target_n = len(kp_names)
    else:
        target_n = int(kps.shape[0])
        kp_names = [f"kp_{i}" for i in range(target_n)]

    if kps.shape[0] < target_n:
        pad = np.zeros((target_n - kps.shape[0], 3), dtype=np.float64)
        kps = np.concatenate([kps, pad], axis=0)
        print(f"[info] Label has fewer keypoints than YAML ({kps.shape[0] - pad.shape[0]} < {target_n}), padded missing as hidden.")
    elif kps.shape[0] > target_n:
        extra = kps.shape[0] - target_n
        kp_names.extend([f"kp_{target_n + i}" for i in range(extra)])
        target_n = kps.shape[0]
    out_label_path = out_label if out_label is not None else label_path

    state = EditorState(
        image=img_bgr,
        label_lines=lines,
        label_line_idx=line_idx,
        class_id=cls_id,
        bbox_xyxy=bbox.copy(),
        kps_xyv=kps.copy(),
        kp_names=kp_names,
        pick_radius=float(pick_radius),
        window_name=window_name,
        out_label=out_label_path,
    )
    state.orig_bbox_xyxy = bbox.copy()
    state.orig_kps_xyv = kps.copy()

    cv2.namedWindow(state.window_name, cv2.WINDOW_NORMAL)
    preview_h = h + (state.hud_height if state.show_hud else 0)
    cv2.resizeWindow(state.window_name, max(1000, w), max(700, preview_h))
    cv2.setMouseCallback(state.window_name, _build_mouse_cb(state))

    print(f"[open] image={image_path}")
    print(f"[open] label={label_path} (line_index={line_index} -> line={line_idx})")
    if yaml_path is not None:
        print(f"[open] yaml={yaml_path}")
    print("[help] n/p: select keypoint | v: toggle visible | b/k: bbox/keypoint mode | s: save | i: HUD | z/x: prev/next image | q: quit")

    while True:
        if cv2.getWindowProperty(state.window_name, cv2.WND_PROP_VISIBLE) < 1:
            if state.dirty:
                print("[warn] Window closed with unsaved changes.")
            cv2.destroyAllWindows()
            return "quit"
        canvas = _draw(state)
        cv2.imshow(state.window_name, canvas)
        key = cv2.waitKey(20) & 0xFF

        if key in (27, ord("q")):
            if state.dirty:
                print("[warn] Unsaved changes. Press 's' to save before quit.")
            else:
                cv2.destroyAllWindows()
                return "quit"
        if key == ord("s"):
            _save(state)
        elif key == ord("x"):
            if state.dirty:
                print("[warn] Unsaved changes. Press 's' before next image.")
            else:
                cv2.destroyAllWindows()
                return "next"
        elif key == ord("z"):
            if state.dirty:
                print("[warn] Unsaved changes. Press 's' before previous image.")
            else:
                cv2.destroyAllWindows()
                return "prev"
        elif key == ord("n") or key == ord("]"):
            state.selected_idx = (state.selected_idx + 1) % max(1, state.kps_xyv.shape[0])
        elif key == ord("p") or key == ord("["):
            state.selected_idx = (state.selected_idx - 1) % max(1, state.kps_xyv.shape[0])
        elif key == ord("v"):
            i = state.selected_idx
            state.kps_xyv[i, 2] = 0.0 if state.kps_xyv[i, 2] > 0.5 else 1.0
            state.dirty = True
        elif key == ord("h"):
            i = state.selected_idx
            state.kps_xyv[i, 2] = 0.0
            state.dirty = True
        elif key == ord("b"):
            state.mode = "bbox"
        elif key == ord("k"):
            state.mode = "keypoint"
        elif key == ord("r"):
            state.bbox_xyxy = state.orig_bbox_xyxy.copy() if state.orig_bbox_xyxy is not None else state.bbox_xyxy
            state.kps_xyv = state.orig_kps_xyv.copy() if state.orig_kps_xyv is not None else state.kps_xyv
            state.dirty = True
        elif key == ord("i"):
            state.show_hud = not state.show_hud

    cv2.destroyAllWindows()
    return "quit"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Interactive pixel-level editor for YOLO pose labels.")
    src = ap.add_mutually_exclusive_group(required=False)
    src.add_argument("--image", type=str, help="Path to one image file")
    src.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help=f"Folder with images to edit sequentially (default: {DEFAULT_IMAGE_DIR})",
    )
    ap.add_argument("--recursive", action="store_true", help="Recursively scan --image-dir for images")
    ap.add_argument("--start-index", type=int, default=0, help="Start image index in folder mode")
    ap.add_argument(
        "--label",
        type=str,
        default=None,
        help="Path to label .txt (single-image mode only, auto-inferred if omitted)",
    )
    ap.add_argument(
        "--yaml",
        type=str,
        default=None,
        help=f"Path to aircraft_pose.yaml (default: {DEFAULT_YAML_PATH}, else auto-inferred)",
    )
    ap.add_argument(
        "--out-label",
        type=str,
        default=None,
        help="Where to save edited label (single-image mode only, defaults to --label path)",
    )
    ap.add_argument(
        "--line-index",
        type=int,
        default=0,
        help="If label has multiple non-empty lines, choose which object to edit",
    )
    ap.add_argument("--pick-radius", type=float, default=18.0, help="Mouse pick radius in pixels")
    ap.add_argument("--window-name", type=str, default="YOLO Pose Pixel Adjuster", help="OpenCV window title")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    yaml_path = Path(args.yaml).expanduser() if args.yaml else (DEFAULT_YAML_PATH if DEFAULT_YAML_PATH.exists() else None)

    image_arg = Path(args.image).expanduser() if args.image else None
    image_dir_arg = Path(args.image_dir).expanduser() if args.image_dir else None
    if image_arg is None and image_dir_arg is None:
        image_dir_arg = DEFAULT_IMAGE_DIR
        print(f"[default] Using image dir: {image_dir_arg}")
        if yaml_path is not None:
            print(f"[default] Using yaml: {yaml_path}")

    if image_dir_arg is not None:
        if args.label or args.out_label:
            raise ValueError("--label/--out-label are only supported with --image (single-image mode).")
        images = _collect_images(image_dir_arg, recursive=bool(args.recursive))
        idx = int(max(0, min(args.start_index, len(images) - 1)))
        while 0 <= idx < len(images):
            print(f"[folder] {idx + 1}/{len(images)}: {images[idx]}")
            action = run(
                image_path=images[idx],
                label_path=None,
                yaml_path=yaml_path,
                out_label=None,
                line_index=int(args.line_index),
                pick_radius=float(args.pick_radius),
                window_name=str(args.window_name),
            )
            if action == "next":
                idx += 1
                if idx >= len(images):
                    print("[done] Reached end of folder.")
                    break
            elif action == "prev":
                idx -= 1
                if idx < 0:
                    print("[done] Reached beginning of folder.")
                    break
            else:
                break
    else:
        run(
            image_path=image_arg,
            label_path=Path(args.label).expanduser() if args.label else None,
            yaml_path=yaml_path,
            out_label=Path(args.out_label).expanduser() if args.out_label else None,
            line_index=int(args.line_index),
            pick_radius=float(args.pick_radius),
            window_name=str(args.window_name),
        )


if __name__ == "__main__":
    main()
