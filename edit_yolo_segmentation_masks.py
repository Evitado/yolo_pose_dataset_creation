#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interactive YOLO-seg mask editor for dataset folders.

Expected dataset layout:
  <dataset_root>/
    images/train|val|test/*.png|*.jpg|...
    labels/train|val|test/*.txt
    *.yaml (optional, for class names)

YOLO-seg label format per line:
  <class_id> x1 y1 x2 y2 x3 y3 ...    (normalized 0..1 polygon points)

Controls:
  - Left click on vertex: select + drag vertex
  - Left click inside polygon: select polygon
  - Right click on selected polygon edge: insert vertex

  - Tab: cycle selected polygon
  - c: cycle class id of selected polygon
  - d: delete selected vertex (if polygon has >3 vertices)
  - x: delete selected polygon
  - a: toggle add mode
      * In add mode: left click to append points
      * Enter: finalize new polygon (>=3 points)
      * Esc: cancel new polygon
      * 0..9: set class id for new polygon

  - s: save labels for current image
  - n / Space: next image (auto-save if dirty)
  - b: previous image (auto-save if dirty)
  - f: toggle fill overlay
  - q: quit
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml


DEFAULT_DATASET_ROOT = Path(
    "/home/femi/yolo_pose_dataset_creation/warning_box_seg_masks/yolo_seg_dataset"
)
DEFAULT_SPLIT = "val"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class PolyObj:
    cls_id: int
    pts_px: np.ndarray  # (N, 2) float32 in pixels


@dataclass
class SceneItem:
    split: str
    image_path: Path
    label_path: Path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Visualize and edit YOLO segmentation polygons.")
    ap.add_argument("--dataset-root", type=str, default=str(DEFAULT_DATASET_ROOT))
    ap.add_argument("--split", type=str, default=DEFAULT_SPLIT, choices=["train", "val", "test", "all"])
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--max-images", type=int, default=0, help="0 = all")
    ap.add_argument("--window", type=str, default="YOLO Segmentation Mask Editor")
    ap.add_argument("--point-radius", type=int, default=5)
    ap.add_argument("--hit-radius", type=int, default=12)
    ap.add_argument("--autosave-nav", type=int, default=1, choices=[0, 1])
    ap.add_argument("--alpha", type=float, default=0.35, help="Fill alpha")
    ap.add_argument("--yaml-path", type=str, default="", help="Optional dataset yaml path for class names")
    return ap.parse_args()


def _load_class_names(dataset_root: Path, yaml_path: str) -> List[str]:
    yaml_candidates: List[Path] = []
    if yaml_path:
        yaml_candidates.append(Path(yaml_path).expanduser().resolve())
    else:
        yaml_candidates.extend(sorted(dataset_root.glob("*.yaml")))
    for yp in yaml_candidates:
        try:
            data = yaml.safe_load(yp.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                continue
            names = data.get("names", None)
            if isinstance(names, dict):
                out: List[str] = []
                for k in sorted(names.keys(), key=lambda x: int(x)):
                    out.append(str(names[k]))
                if out:
                    return out
            if isinstance(names, list) and names:
                return [str(x) for x in names]
        except Exception:
            continue
    return ["aircraft", "engine_left", "engine_right", "front_gear"]


def _class_name(class_names: Sequence[str], cls_id: int) -> str:
    if 0 <= int(cls_id) < len(class_names):
        return str(class_names[int(cls_id)])
    return f"class_{int(cls_id)}"


def _infer_aircraft_class_ids(class_names: Sequence[str]) -> List[int]:
    ids: List[int] = []
    for i, name in enumerate(class_names):
        n = str(name).strip().lower()
        if ("aircraft" in n) or (n == "plane") or ("fuselage" in n):
            ids.append(int(i))
    return ids


def _collect_items(dataset_root: Path, split: str) -> List[SceneItem]:
    splits = ["train", "val", "test"] if split == "all" else [split]
    items: List[SceneItem] = []
    for sp in splits:
        img_dir = dataset_root / "images" / sp
        lbl_dir = dataset_root / "labels" / sp
        if not img_dir.exists():
            continue
        for ip in sorted(img_dir.iterdir()):
            if not ip.is_file() or ip.suffix.lower() not in IMAGE_EXTS:
                continue
            lp = lbl_dir / f"{ip.stem}.txt"
            items.append(SceneItem(split=sp, image_path=ip, label_path=lp))
    return items


def _parse_seg_labels(label_path: Path, w: int, h: int) -> List[PolyObj]:
    polys: List[PolyObj] = []
    if not label_path.exists():
        return polys
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return polys
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split()
        if len(parts) < 7:
            continue
        try:
            cls_id = int(float(parts[0]))
            vals = [float(x) for x in parts[1:]]
        except Exception:
            continue
        if len(vals) < 6 or (len(vals) % 2) != 0:
            continue
        arr = np.asarray(vals, dtype=np.float32).reshape(-1, 2)
        arr[:, 0] = np.clip(arr[:, 0], 0.0, 1.0) * float(w - 1)
        arr[:, 1] = np.clip(arr[:, 1], 0.0, 1.0) * float(h - 1)
        if arr.shape[0] >= 3:
            polys.append(PolyObj(cls_id=cls_id, pts_px=arr.astype(np.float32)))
    return polys


def _write_seg_labels(label_path: Path, polys: Sequence[PolyObj], w: int, h: int) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    out_lines: List[str] = []
    sx = max(1.0, float(w - 1))
    sy = max(1.0, float(h - 1))
    for p in polys:
        pts = np.asarray(p.pts_px, dtype=np.float32).reshape(-1, 2)
        if pts.shape[0] < 3:
            continue
        nx = np.clip(pts[:, 0] / sx, 0.0, 1.0)
        ny = np.clip(pts[:, 1] / sy, 0.0, 1.0)
        flat: List[str] = []
        for x, y in zip(nx.tolist(), ny.tolist()):
            flat.append(f"{float(x):.6f}")
            flat.append(f"{float(y):.6f}")
        out_lines.append(f"{int(p.cls_id)} " + " ".join(flat))
    txt = "\n".join(out_lines)
    if txt:
        txt += "\n"
    label_path.write_text(txt, encoding="utf-8")


def _palette(i: int) -> Tuple[int, int, int]:
    base = [
        (0, 255, 0),
        (255, 200, 0),
        (0, 200, 255),
        (255, 0, 255),
        (80, 180, 255),
        (255, 120, 120),
        (120, 255, 160),
        (180, 120, 255),
    ]
    return base[int(i) % len(base)]


def _closest_vertex(
    polys: Sequence[PolyObj],
    qx: float,
    qy: float,
    max_dist_px: float,
) -> Tuple[Optional[int], Optional[int], float]:
    best_poly = None
    best_v = None
    best_d2 = float(max_dist_px * max_dist_px)
    for pi, poly in enumerate(polys):
        pts = np.asarray(poly.pts_px, dtype=np.float32)
        if pts.size == 0:
            continue
        d2 = (pts[:, 0] - float(qx)) ** 2 + (pts[:, 1] - float(qy)) ** 2
        vi = int(np.argmin(d2))
        if float(d2[vi]) <= best_d2:
            best_d2 = float(d2[vi])
            best_poly = int(pi)
            best_v = int(vi)
    return best_poly, best_v, math.sqrt(best_d2) if best_poly is not None else float("inf")


def _point_in_poly_idx(polys: Sequence[PolyObj], x: int, y: int) -> Optional[int]:
    for pi in reversed(range(len(polys))):
        pts = np.asarray(polys[pi].pts_px, dtype=np.float32).reshape(-1, 1, 2)
        if pts.shape[0] < 3:
            continue
        inside = cv2.pointPolygonTest(pts, (float(x), float(y)), False)
        if inside >= 0:
            return int(pi)
    return None


def _closest_edge_insert(
    pts: np.ndarray,
    q: np.ndarray,
    max_dist_px: float,
) -> Tuple[Optional[int], Optional[np.ndarray]]:
    n = int(pts.shape[0])
    if n < 2:
        return None, None
    best_i = None
    best_proj = None
    best_d2 = float(max_dist_px * max_dist_px)
    for i in range(n):
        a = pts[i]
        b = pts[(i + 1) % n]
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom <= 1e-9:
            continue
        t = float(np.dot(q - a, ab) / denom)
        t = max(0.0, min(1.0, t))
        proj = a + t * ab
        d2 = float(np.dot(q - proj, q - proj))
        if d2 <= best_d2:
            best_d2 = d2
            best_i = int(i)
            best_proj = proj
    return best_i, best_proj


class Editor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.dataset_root = Path(args.dataset_root).expanduser().resolve()
        self.class_names = _load_class_names(self.dataset_root, str(args.yaml_path or ""))
        self.aircraft_cls_ids = set(_infer_aircraft_class_ids(self.class_names))
        self.items = _collect_items(self.dataset_root, str(args.split))
        if args.start_index > 0:
            self.items = self.items[int(args.start_index):]
        if int(args.max_images) > 0:
            self.items = self.items[: int(args.max_images)]
        if not self.items:
            raise RuntimeError(f"No images found under {self.dataset_root} split={args.split}")

        self.idx = 0
        self.img_bgr: Optional[np.ndarray] = None
        self.h = 0
        self.w = 0
        self.polys: List[PolyObj] = []
        self.sel_poly: Optional[int] = None
        self.sel_vertex: Optional[int] = None
        self.dragging = False
        self.drag_poly_idx: Optional[int] = None
        self.drag_vertex_idx: Optional[int] = None
        self.dirty = False
        self.show_fill = True
        self.alpha = float(np.clip(float(args.alpha), 0.0, 1.0))

        self.add_mode = False
        self.add_cls_id = 0
        self.add_pts: List[Tuple[float, float]] = []

        self.window = str(args.window)
        self._load_current()

    @property
    def cur(self) -> SceneItem:
        return self.items[self.idx]

    def _load_current(self) -> None:
        bgr = cv2.imread(str(self.cur.image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to read image: {self.cur.image_path}")
        self.img_bgr = bgr
        self.h, self.w = bgr.shape[:2]
        self.polys = _parse_seg_labels(self.cur.label_path, self.w, self.h)
        self.sel_poly = 0 if self.polys else None
        self.sel_vertex = None
        self.dragging = False
        self.drag_poly_idx = None
        self.drag_vertex_idx = None
        self.dirty = False
        self.add_mode = False
        self.add_pts = []
        if self.sel_poly is not None:
            self.add_cls_id = int(self.polys[self.sel_poly].cls_id)

    def _save(self) -> None:
        _write_seg_labels(self.cur.label_path, self.polys, self.w, self.h)
        self.dirty = False
        print(f"[saved] {self.cur.label_path}")

    def _clip_xy(self, x: float, y: float) -> Tuple[float, float]:
        xx = float(np.clip(x, 0.0, float(self.w - 1)))
        yy = float(np.clip(y, 0.0, float(self.h - 1)))
        return xx, yy

    def _render_order_indices(self) -> List[int]:
        aircraft: List[int] = []
        others: List[int] = []
        for i, p in enumerate(self.polys):
            if int(p.cls_id) in self.aircraft_cls_ids:
                aircraft.append(int(i))
            else:
                others.append(int(i))
        # Aircraft first (underneath), others later (on top).
        return aircraft + others

    def _point_in_poly_idx_view(self, x: int, y: int) -> Optional[int]:
        draw_order = self._render_order_indices()
        for pi in reversed(draw_order):
            pts = np.asarray(self.polys[int(pi)].pts_px, dtype=np.float32).reshape(-1, 1, 2)
            if pts.shape[0] < 3:
                continue
            inside = cv2.pointPolygonTest(pts, (float(x), float(y)), False)
            if inside >= 0:
                return int(pi)
        return None

    def _render(self) -> np.ndarray:
        assert self.img_bgr is not None
        out = self.img_bgr.copy()
        overlay = out.copy()

        for pi in self._render_order_indices():
            p = self.polys[int(pi)]
            pts = np.asarray(p.pts_px, dtype=np.float32).reshape(-1, 1, 2).astype(np.int32)
            if pts.shape[0] < 3:
                continue
            col = _palette(int(p.cls_id))
            if self.show_fill:
                cv2.fillPoly(overlay, [pts], col)
            thick = 3 if (self.sel_poly is not None and int(pi) == int(self.sel_poly)) else 2
            cv2.polylines(out, [pts], True, col, thick, lineType=cv2.LINE_AA)

            p0 = pts[0, 0]
            name = _class_name(self.class_names, int(p.cls_id))
            cv2.putText(
                out,
                f"{pi}:{name}",
                (int(p0[0]) + 4, int(p0[1]) - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                col,
                1,
                cv2.LINE_AA,
            )

            for vi, v in enumerate(np.asarray(p.pts_px, dtype=np.float32)):
                r = int(self.args.point_radius)
                if self.sel_poly is not None and self.sel_vertex is not None:
                    if int(pi) == int(self.sel_poly) and int(vi) == int(self.sel_vertex):
                        cv2.circle(out, (int(v[0]), int(v[1])), r + 2, (255, 255, 255), -1, cv2.LINE_AA)
                        cv2.circle(out, (int(v[0]), int(v[1])), r + 1, (0, 0, 255), 2, cv2.LINE_AA)
                        continue
                cv2.circle(out, (int(v[0]), int(v[1])), r, col, -1, cv2.LINE_AA)

        if self.show_fill:
            out = cv2.addWeighted(overlay, self.alpha, out, 1.0 - self.alpha, 0.0)

        if self.add_mode and self.add_pts:
            pts_arr = np.asarray(self.add_pts, dtype=np.float32).reshape(-1, 1, 2).astype(np.int32)
            col = _palette(self.add_cls_id)
            if len(self.add_pts) >= 2:
                cv2.polylines(out, [pts_arr], False, col, 2, lineType=cv2.LINE_AA)
            for p in self.add_pts:
                cv2.circle(out, (int(p[0]), int(p[1])), int(self.args.point_radius), col, -1, cv2.LINE_AA)

        status = (
            f"{self.idx+1}/{len(self.items)} split={self.cur.split} "
            f"img={self.cur.image_path.name} polys={len(self.polys)} dirty={int(self.dirty)}"
        )
        cv2.rectangle(out, (0, 0), (self.w, 52), (0, 0, 0), -1)
        cv2.putText(out, status, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)

        if self.add_mode:
            add_name = _class_name(self.class_names, self.add_cls_id)
            msg = f"ADD MODE class={self.add_cls_id}:{add_name} points={len(self.add_pts)} (Enter finalize, Esc cancel)"
        else:
            msg = "L-drag vertex, R-click edge add-vertex, Tab cycle poly, c class, d del-vertex, x del-poly"
        cv2.putText(out, msg, (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 200), 1, cv2.LINE_AA)
        return out

    def _select_poly_safe(self) -> None:
        if not self.polys:
            self.sel_poly = None
            self.sel_vertex = None
            return
        if self.sel_poly is None:
            self.sel_poly = 0
            self.sel_vertex = None
            return
        if int(self.sel_poly) < 0 or int(self.sel_poly) >= len(self.polys):
            self.sel_poly = len(self.polys) - 1
            self.sel_vertex = None

    def _on_mouse(self, event: int, x: int, y: int, flags: int) -> None:
        _ = flags
        if self.img_bgr is None:
            return

        if self.add_mode:
            if event == cv2.EVENT_LBUTTONDOWN:
                xx, yy = self._clip_xy(float(x), float(y))
                self.add_pts.append((xx, yy))
                self.dirty = True
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            p_idx, v_idx, _d = _closest_vertex(
                self.polys,
                float(x),
                float(y),
                float(self.args.hit_radius),
            )
            if p_idx is not None and v_idx is not None:
                self.sel_poly = int(p_idx)
                self.sel_vertex = int(v_idx)
                self.dragging = True
                self.drag_poly_idx = int(p_idx)
                self.drag_vertex_idx = int(v_idx)
                return
            inside_idx = self._point_in_poly_idx_view(int(x), int(y))
            if inside_idx is not None:
                self.sel_poly = int(inside_idx)
                self.sel_vertex = None
                return

        if event == cv2.EVENT_MOUSEMOVE and self.dragging:
            if self.drag_poly_idx is None or self.drag_vertex_idx is None:
                return
            if not (0 <= int(self.drag_poly_idx) < len(self.polys)):
                return
            poly = self.polys[int(self.drag_poly_idx)]
            if not (0 <= int(self.drag_vertex_idx) < poly.pts_px.shape[0]):
                return
            xx, yy = self._clip_xy(float(x), float(y))
            poly.pts_px[int(self.drag_vertex_idx), 0] = float(xx)
            poly.pts_px[int(self.drag_vertex_idx), 1] = float(yy)
            self.dirty = True
            return

        if event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
            self.drag_poly_idx = None
            self.drag_vertex_idx = None
            return

        if event == cv2.EVENT_RBUTTONDOWN:
            if self.sel_poly is None:
                return
            if not (0 <= int(self.sel_poly) < len(self.polys)):
                return
            poly = self.polys[int(self.sel_poly)]
            pts = np.asarray(poly.pts_px, dtype=np.float32).reshape(-1, 2)
            if pts.shape[0] < 2:
                return
            edge_i, proj = _closest_edge_insert(
                pts,
                np.asarray([float(x), float(y)], dtype=np.float32),
                float(self.args.hit_radius),
            )
            if edge_i is None or proj is None:
                return
            insert_at = int(edge_i) + 1
            poly.pts_px = np.insert(poly.pts_px, insert_at, proj.reshape(1, 2), axis=0)
            self.sel_vertex = int(insert_at)
            self.dirty = True
            return

    def _go_to(self, new_idx: int) -> None:
        if new_idx < 0 or new_idx >= len(self.items):
            return
        if self.dirty and int(self.args.autosave_nav) == 1:
            self._save()
        self.idx = int(new_idx)
        self._load_current()

    def run(self) -> None:
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window, 1400, 860)

        def _mouse_cb(event: int, x: int, y: int, flags: int, param: object) -> None:
            _ = param
            self._on_mouse(event, x, y, flags)

        cv2.setMouseCallback(self.window, _mouse_cb)

        while True:
            canvas = self._render()
            cv2.imshow(self.window, canvas)
            key = cv2.waitKey(20) & 0xFF
            if key == 255:
                continue

            if key == ord("q"):
                if self.dirty:
                    self._save()
                break

            if key in (ord("n"), ord(" ")):
                self._go_to(self.idx + 1)
                continue
            if key == ord("b"):
                self._go_to(self.idx - 1)
                continue

            if key == ord("s"):
                self._save()
                continue

            if key == ord("f"):
                self.show_fill = not bool(self.show_fill)
                continue

            if key == 9:  # tab
                if self.polys:
                    if self.sel_poly is None:
                        self.sel_poly = 0
                    else:
                        self.sel_poly = (int(self.sel_poly) + 1) % len(self.polys)
                    self.sel_vertex = None
                continue

            if key == ord("x"):
                if self.sel_poly is not None and 0 <= int(self.sel_poly) < len(self.polys):
                    del self.polys[int(self.sel_poly)]
                    self._select_poly_safe()
                    self.dirty = True
                continue

            if key == ord("d"):
                if (
                    self.sel_poly is not None
                    and self.sel_vertex is not None
                    and 0 <= int(self.sel_poly) < len(self.polys)
                ):
                    poly = self.polys[int(self.sel_poly)]
                    if poly.pts_px.shape[0] > 3 and 0 <= int(self.sel_vertex) < poly.pts_px.shape[0]:
                        poly.pts_px = np.delete(poly.pts_px, int(self.sel_vertex), axis=0)
                        self.sel_vertex = None
                        self.dirty = True
                continue

            if key == ord("c"):
                if self.sel_poly is not None and 0 <= int(self.sel_poly) < len(self.polys):
                    ncls = max(1, len(self.class_names))
                    self.polys[int(self.sel_poly)].cls_id = (
                        int(self.polys[int(self.sel_poly)].cls_id) + 1
                    ) % ncls
                    self.dirty = True
                elif self.add_mode:
                    ncls = max(1, len(self.class_names))
                    self.add_cls_id = (int(self.add_cls_id) + 1) % ncls
                continue

            if key == ord("a"):
                if not self.add_mode:
                    self.add_mode = True
                    self.add_pts = []
                    if self.sel_poly is not None and 0 <= int(self.sel_poly) < len(self.polys):
                        self.add_cls_id = int(self.polys[int(self.sel_poly)].cls_id)
                    else:
                        self.add_cls_id = 0
                else:
                    self.add_mode = False
                    self.add_pts = []
                continue

            if key == 13:  # Enter
                if self.add_mode and len(self.add_pts) >= 3:
                    arr = np.asarray(self.add_pts, dtype=np.float32).reshape(-1, 2)
                    self.polys.append(PolyObj(cls_id=int(self.add_cls_id), pts_px=arr))
                    self.sel_poly = len(self.polys) - 1
                    self.sel_vertex = None
                    self.add_mode = False
                    self.add_pts = []
                    self.dirty = True
                continue

            if key == 27:  # Esc
                if self.add_mode:
                    self.add_mode = False
                    self.add_pts = []
                continue

            if self.add_mode and ord("0") <= key <= ord("9"):
                self.add_cls_id = int(key - ord("0"))
                continue

        cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    editor = Editor(args)
    print(
        f"[editor] dataset={editor.dataset_root} split={args.split} "
        f"images={len(editor.items)} classes={editor.class_names}"
    )
    editor.run()


if __name__ == "__main__":
    main()
