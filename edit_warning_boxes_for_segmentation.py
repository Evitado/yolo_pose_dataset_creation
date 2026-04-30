#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interactive warning-box editor for segmentation correction.

Workflow:
1) Show 2D segmentation overlay for a scene.
2) If wrong, press 'e' to open a 3D warning-box editor.
3) Move/rotate/scale warning boxes in 3D.
4) Save edits to JSON.
5) Regenerate masks using create_warning_box_segmentation_images.py with --edits-json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

import create_warning_box_segmentation_images as wbseg


DEFAULT_IMAGE_DIR = (
    "/home/femi/yolo_pose_dataset_creation/warning_box_seg_masks/yolo_seg_dataset/images/val"
)
DEFAULT_SOURCE_H5_ROOT = "/home/femi/Benchmarking_framework/Data/warning_b_test_h5"
DEFAULT_EDITS_JSON = "/home/femi/yolo_pose_dataset_creation/warning_box_edits.json"
DEFAULT_PREVIEW_OUT = "/home/femi/yolo_pose_dataset_creation/warning_box_edit_preview"
DEFAULT_YOLO_DATASET_DIR = "/home/femi/yolo_pose_dataset_creation/warning_box_seg_masks/yolo_seg_dataset"


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Interactive 2D+3D warning-box editor for segmentation masks."
    )
    ap.add_argument("--image-dir", type=str, default=DEFAULT_IMAGE_DIR)
    ap.add_argument("--source", type=str, default=DEFAULT_SOURCE_H5_ROOT)
    ap.add_argument("--edits-json", type=str, default=DEFAULT_EDITS_JSON)
    ap.add_argument("--preview-out", type=str, default=DEFAULT_PREVIEW_OUT)
    ap.add_argument(
        "--class-map",
        type=str,
        default="front_landing_gear:1,engine_left:2,engine_right:3",
    )
    ap.add_argument("--include-aircraft-mask", type=int, default=1, choices=[0, 1])
    ap.add_argument(
        "--show-aircraft-mask",
        type=int,
        default=0,
        choices=[0, 1],
        help="Show aircraft class in 2D editor overlay (default: hidden).",
    )
    ap.add_argument("--aircraft-class-id", type=int, default=4)
    ap.add_argument("--overlay-alpha", type=float, default=0.45)
    ap.add_argument("--warning-box-scale", type=float, default=1.0)
    ap.add_argument("--warning-profile-csv", type=str, default="")
    ap.add_argument("--warning-yaml-column", type=str, default="recommended_yaml")
    ap.add_argument("--warning-yaml-root", type=str, default="")
    ap.add_argument("--warning-yaml-relpath", type=str, default="detection_configs/default.yaml")
    ap.add_argument("--warning-target-level", type=int, default=5)
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--point-limit", type=int, default=250000, help="Max points shown in 3D view")
    ap.add_argument("--trans-step", type=float, default=0.10, help="Translation step in meters")
    ap.add_argument("--rot-step-deg", type=float, default=1.0, help="Rotation step in degrees")
    ap.add_argument("--scale-step", type=float, default=0.05, help="Scale step ratio")
    ap.add_argument(
        "--auto-update-yolo-labels",
        type=int,
        default=1,
        choices=[0, 1],
        help="Write YOLO segmentation labels immediately when scene edits change.",
    )
    ap.add_argument(
        "--yolo-dataset-dir",
        type=str,
        default=DEFAULT_YOLO_DATASET_DIR,
        help="YOLO dataset root containing images/ and labels/ folders.",
    )
    ap.add_argument(
        "--yolo-split",
        type=str,
        default="auto",
        choices=["auto", "train", "val", "test"],
        help="Split for YOLO label writing. 'auto' infers from image path.",
    )
    ap.add_argument(
        "--min-contour-area",
        type=float,
        default=25.0,
        help="Minimum contour area in pixels when writing YOLO polygons.",
    )
    ap.add_argument(
        "--contour-approx-eps",
        type=float,
        default=1.0,
        help="Polygon approximation epsilon in pixels when writing YOLO polygons.",
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
        default=wbseg.DEFAULT_A380_DET_DATASET_ROOT,
        help="YOLO detection dataset root containing labels/train|val|test/*.txt.",
    )
    ap.add_argument(
        "--a380-det-front-class-id",
        type=int,
        default=3,
        help="YOLO detection class id for front_gear bbox.",
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


def _deepcopy_specs(specs: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for k, v in specs.items():
        out[str(k)] = {
            "source_name": str(v.get("source_name", "")),
            "center_world": np.asarray(v["center_world"], dtype=np.float64).reshape(3).copy(),
            "half": np.asarray(v["half"], dtype=np.float64).reshape(3).copy(),
            "R_world_to_box": np.asarray(v["R_world_to_box"], dtype=np.float64).reshape(3, 3).copy(),
        }
    return out


def _spec_to_jsonable(spec: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "source_name": str(spec.get("source_name", "")),
        "center_world": np.asarray(spec["center_world"], dtype=np.float64).reshape(3).tolist(),
        "half": np.asarray(spec["half"], dtype=np.float64).reshape(3).tolist(),
        "R_world_to_box": np.asarray(spec["R_world_to_box"], dtype=np.float64).reshape(3, 3).tolist(),
    }


def _specs_equal(a: Dict[str, Dict[str, Any]], b: Dict[str, Dict[str, Any]]) -> bool:
    a_keys = set(a.keys())
    b_keys = set(b.keys())
    if a_keys != b_keys:
        return False
    for k in sorted(a_keys):
        aa = a[k]
        bb = b[k]
        if not np.allclose(np.asarray(aa["center_world"]), np.asarray(bb["center_world"]), atol=1e-9):
            return False
        if not np.allclose(np.asarray(aa["half"]), np.asarray(bb["half"]), atol=1e-9):
            return False
        if not np.allclose(np.asarray(aa["R_world_to_box"]), np.asarray(bb["R_world_to_box"]), atol=1e-9):
            return False
    return True


def _bag_prefix_from_unique_scene(unique_scene: str) -> str:
    s = str(unique_scene)
    if "__scene_" in s:
        return s.rsplit("__scene_", 1)[0]
    parts = s.rsplit("__", 1)
    if len(parts) == 2 and parts[1].startswith("scene_"):
        return parts[0]
    return s


def _orthonormalize_rotation(R: np.ndarray) -> np.ndarray:
    M = np.asarray(R, dtype=np.float64).reshape(3, 3)
    U, _, Vt = np.linalg.svd(M)
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1.0
        Rn = U @ Vt
    return Rn


def _compute_delta_map(
    base_specs: Dict[str, Dict[str, Any]],
    cur_specs: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, np.ndarray]]:
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for k in sorted(set(base_specs.keys()) & set(cur_specs.keys())):
        b = base_specs[k]
        c = cur_specs[k]
        cb = np.asarray(b["center_world"], dtype=np.float64).reshape(3)
        cc = np.asarray(c["center_world"], dtype=np.float64).reshape(3)
        hb = np.asarray(b["half"], dtype=np.float64).reshape(3)
        hc = np.asarray(c["half"], dtype=np.float64).reshape(3)
        Rb = np.asarray(b["R_world_to_box"], dtype=np.float64).reshape(3, 3)
        Rc = np.asarray(c["R_world_to_box"], dtype=np.float64).reshape(3, 3)
        denom = np.maximum(np.abs(hb), 1e-9)
        hs = np.maximum(hc / denom, 1e-6)
        Rdelta = _orthonormalize_rotation(Rb.T @ Rc)
        out[str(k)] = {
            "dc": (cc - cb),
            "hs": hs,
            "Rdelta": Rdelta,
        }
    return out


def _apply_delta_map_to_specs(
    specs_base: Dict[str, Dict[str, Any]],
    delta_map: Dict[str, Dict[str, np.ndarray]],
) -> Dict[str, Dict[str, Any]]:
    out = _deepcopy_specs(specs_base)
    for k, delta in delta_map.items():
        if k not in out:
            continue
        s = out[k]
        c = np.asarray(s["center_world"], dtype=np.float64).reshape(3)
        h = np.asarray(s["half"], dtype=np.float64).reshape(3)
        R = np.asarray(s["R_world_to_box"], dtype=np.float64).reshape(3, 3)
        c_new = c + np.asarray(delta["dc"], dtype=np.float64).reshape(3)
        h_new = np.maximum(h * np.asarray(delta["hs"], dtype=np.float64).reshape(3), 1e-6)
        R_new = _orthonormalize_rotation(R @ np.asarray(delta["Rdelta"], dtype=np.float64).reshape(3, 3))
        s["center_world"] = c_new
        s["half"] = h_new
        s["R_world_to_box"] = R_new
    return out


def _save_edits_json(path: Path, edits_data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    txt = json.dumps(edits_data, indent=2, sort_keys=True)
    path.write_text(txt + "\n", encoding="utf-8")
    print(f"[saved] edits json: {path} (scenes={len(edits_data)})")


def _axis_rotation(axis: str, deg: float) -> np.ndarray:
    th = float(np.deg2rad(deg))
    c, s = float(np.cos(th)), float(np.sin(th))
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def _box_color(box_key: str) -> Tuple[float, float, float]:
    k = str(box_key)
    if "front" in k:
        return (1.0, 0.0, 0.0)
    if "left" in k:
        return (0.0, 1.0, 0.0)
    if "right" in k:
        return (1.0, 0.6, 0.0)
    return (0.8, 0.8, 0.8)


def _build_lineset_from_spec(o3d: Any, pcd_view: Any, spec: Dict[str, Any], color: Tuple[float, float, float]) -> Any:
    c = np.asarray(spec["center_world"], dtype=np.float64).reshape(3)
    h = np.asarray(spec["half"], dtype=np.float64).reshape(3)
    Rwb = np.asarray(spec["R_world_to_box"], dtype=np.float64).reshape(3, 3)
    corners = pcd_view._oriented_box_corners(c, h, Rwb)
    edges = np.asarray(pcd_view._warning_edges(), dtype=np.int32).reshape(-1, 2)

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(corners)
    ls.lines = o3d.utility.Vector2iVector(edges)
    col = np.asarray(color, dtype=np.float64).reshape(1, 3)
    ls.colors = o3d.utility.Vector3dVector(np.repeat(col, edges.shape[0], axis=0))
    return ls


def _update_lineset(o3d: Any, pcd_view: Any, ls: Any, spec: Dict[str, Any], color: Tuple[float, float, float]) -> None:
    c = np.asarray(spec["center_world"], dtype=np.float64).reshape(3)
    h = np.asarray(spec["half"], dtype=np.float64).reshape(3)
    Rwb = np.asarray(spec["R_world_to_box"], dtype=np.float64).reshape(3, 3)
    corners = pcd_view._oriented_box_corners(c, h, Rwb)
    edges = np.asarray(pcd_view._warning_edges(), dtype=np.int32).reshape(-1, 2)
    ls.points = o3d.utility.Vector3dVector(corners)
    ls.lines = o3d.utility.Vector2iVector(edges)
    col = np.asarray(color, dtype=np.float64).reshape(1, 3)
    ls.colors = o3d.utility.Vector3dVector(np.repeat(col, edges.shape[0], axis=0))


def _edit_specs_in_3d(
    *,
    xyz_hw3: np.ndarray,
    specs: Dict[str, Dict[str, Any]],
    scene_name: str,
    point_limit: int,
    trans_step: float,
    rot_step_deg: float,
    scale_step: float,
    pcd_view: Any,
) -> Tuple[Dict[str, Dict[str, Any]], bool]:
    try:
        import open3d as o3d
    except Exception as e:
        raise RuntimeError(
            "open3d is required for 3D box editing. Use an env with open3d installed."
        ) from e

    keys = [k for k in ("front_landing_gear", "engine_left", "engine_right") if k in specs]
    keys.extend([k for k in specs.keys() if k not in keys])
    if not keys:
        return specs, False

    specs_edit = _deepcopy_specs(specs)
    selected_idx = 0
    changed = False

    pts = np.asarray(xyz_hw3, dtype=np.float64).reshape(-1, 3)
    finite = np.all(np.isfinite(pts), axis=1)
    pts = pts[finite]
    if pts.shape[0] > int(max(1000, point_limit)):
        rng = np.random.default_rng(1234)
        sel = rng.choice(pts.shape[0], size=int(point_limit), replace=False)
        pts = pts[sel]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.paint_uniform_color([0.65, 0.65, 0.65])

    line_sets: Dict[str, Any] = {}
    for k in keys:
        line_sets[k] = _build_lineset_from_spec(
            o3d, pcd_view, specs_edit[k], _box_color(k)
        )

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(
        window_name=f"Warning Box 3D Editor: {scene_name}",
        width=1400,
        height=900,
    )
    vis.add_geometry(pcd)
    for k in keys:
        vis.add_geometry(line_sets[k])

    print("\n[3d editor controls]")
    print("  Z/X   : select prev/next box")
    print("  9/0   : select prev/next box")
    print("  1/2/3 : select front/left/right")
    print("  I/K   : move +X/-X")
    print("  J/L   : move -Y/+Y")
    print("  U/O   : move -Z/+Z")
    print("  R/F   : yaw +/ - (around box Z)")
    print("  T/G   : pitch +/ - (around box Y)")
    print("  Y/H   : roll +/ - (around box X)")
    print("  =/-   : scale up/down")
    print("  W/S   : scale X (length) +/ -")
    print("  E/D   : scale Y (width) +/ -")
    print("  C/V   : scale Z (height) +/ -")
    print("  P     : print selected box state")
    print("  Q/Esc : close 3D editor\n")

    def _refresh() -> None:
        for i, k in enumerate(keys):
            base = _box_color(k)
            # Keep class colors visible even for the selected box.
            col = base
            _update_lineset(o3d, pcd_view, line_sets[k], specs_edit[k], col)
            vis.update_geometry(line_sets[k])
        vis.update_renderer()

    def _sel_key() -> str:
        return keys[selected_idx]

    def _set_sel_by_name(name: str) -> None:
        nonlocal selected_idx
        if name in keys:
            selected_idx = keys.index(name)
            print(f"[3d] selected: {name}")
            _refresh()

    def _move(dx: float, dy: float, dz: float) -> None:
        nonlocal changed
        k = _sel_key()
        c = np.asarray(specs_edit[k]["center_world"], dtype=np.float64).reshape(3)
        c = c + np.asarray([dx, dy, dz], dtype=np.float64)
        specs_edit[k]["center_world"] = c
        changed = True
        _refresh()

    def _rotate(axis: str, deg: float) -> None:
        nonlocal changed
        k = _sel_key()
        R = np.asarray(specs_edit[k]["R_world_to_box"], dtype=np.float64).reshape(3, 3)
        Rdelta = _axis_rotation(axis, deg)
        # Rotate in box-local frame (right multiplication).
        specs_edit[k]["R_world_to_box"] = R @ Rdelta
        changed = True
        _refresh()

    def _scale(mult: float) -> None:
        nonlocal changed
        k = _sel_key()
        h = np.asarray(specs_edit[k]["half"], dtype=np.float64).reshape(3)
        h = np.maximum(h * float(mult), 1e-3)
        specs_edit[k]["half"] = h
        changed = True
        _refresh()

    def _scale_axis(axis_idx: int, mult: float) -> None:
        nonlocal changed
        k = _sel_key()
        h = np.asarray(specs_edit[k]["half"], dtype=np.float64).reshape(3)
        ai = int(np.clip(int(axis_idx), 0, 2))
        h[ai] = float(max(1e-3, h[ai] * float(mult)))
        specs_edit[k]["half"] = h
        changed = True
        _refresh()

    def _print_state() -> None:
        k = _sel_key()
        s = specs_edit[k]
        c = np.asarray(s["center_world"], dtype=np.float64).reshape(3)
        h = np.asarray(s["half"], dtype=np.float64).reshape(3)
        print(
            f"[3d] {k} center=({c[0]:.3f},{c[1]:.3f},{c[2]:.3f}) "
            f"half=({h[0]:.3f},{h[1]:.3f},{h[2]:.3f})"
        )

    def _cb_select_prev(v: Any) -> bool:
        nonlocal selected_idx
        selected_idx = (selected_idx - 1) % len(keys)
        print(f"[3d] selected: {_sel_key()}")
        _refresh()
        return False

    def _cb_select_next(v: Any) -> bool:
        nonlocal selected_idx
        selected_idx = (selected_idx + 1) % len(keys)
        print(f"[3d] selected: {_sel_key()}")
        _refresh()
        return False

    # Selection
    vis.register_key_callback(ord("Z"), _cb_select_prev)
    vis.register_key_callback(ord("X"), _cb_select_next)
    vis.register_key_callback(ord("9"), _cb_select_prev)
    vis.register_key_callback(ord("0"), _cb_select_next)
    vis.register_key_callback(ord("1"), lambda v: (_set_sel_by_name("front_landing_gear"), False)[1])
    vis.register_key_callback(ord("2"), lambda v: (_set_sel_by_name("engine_left"), False)[1])
    vis.register_key_callback(ord("3"), lambda v: (_set_sel_by_name("engine_right"), False)[1])

    # Translation
    vis.register_key_callback(ord("I"), lambda v: (_move(+trans_step, 0.0, 0.0), False)[1])
    vis.register_key_callback(ord("K"), lambda v: (_move(-trans_step, 0.0, 0.0), False)[1])
    vis.register_key_callback(ord("J"), lambda v: (_move(0.0, -trans_step, 0.0), False)[1])
    vis.register_key_callback(ord("L"), lambda v: (_move(0.0, +trans_step, 0.0), False)[1])
    vis.register_key_callback(ord("U"), lambda v: (_move(0.0, 0.0, -trans_step), False)[1])
    vis.register_key_callback(ord("O"), lambda v: (_move(0.0, 0.0, +trans_step), False)[1])

    # Rotation
    vis.register_key_callback(ord("R"), lambda v: (_rotate("z", +rot_step_deg), False)[1])
    vis.register_key_callback(ord("F"), lambda v: (_rotate("z", -rot_step_deg), False)[1])
    vis.register_key_callback(ord("T"), lambda v: (_rotate("y", +rot_step_deg), False)[1])
    vis.register_key_callback(ord("G"), lambda v: (_rotate("y", -rot_step_deg), False)[1])
    vis.register_key_callback(ord("Y"), lambda v: (_rotate("x", +rot_step_deg), False)[1])
    vis.register_key_callback(ord("H"), lambda v: (_rotate("x", -rot_step_deg), False)[1])

    # Scale and print
    vis.register_key_callback(ord("="), lambda v: (_scale(1.0 + scale_step), False)[1])
    vis.register_key_callback(ord("-"), lambda v: (_scale(max(1e-3, 1.0 - scale_step)), False)[1])
    vis.register_key_callback(ord("W"), lambda v: (_scale_axis(0, 1.0 + scale_step), False)[1])
    vis.register_key_callback(ord("S"), lambda v: (_scale_axis(0, max(1e-3, 1.0 - scale_step)), False)[1])
    vis.register_key_callback(ord("E"), lambda v: (_scale_axis(1, 1.0 + scale_step), False)[1])
    vis.register_key_callback(ord("D"), lambda v: (_scale_axis(1, max(1e-3, 1.0 - scale_step)), False)[1])
    vis.register_key_callback(ord("C"), lambda v: (_scale_axis(2, 1.0 + scale_step), False)[1])
    vis.register_key_callback(ord("V"), lambda v: (_scale_axis(2, max(1e-3, 1.0 - scale_step)), False)[1])
    vis.register_key_callback(ord("P"), lambda v: (_print_state(), False)[1])

    _refresh()
    print(f"[3d] selected: {_sel_key()}")
    vis.run()
    vis.destroy_window()
    return specs_edit, bool(changed)


class Session:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.image_dir = Path(str(args.image_dir)).expanduser().resolve()
        self.source_root = str(args.source)
        self.edits_json_path = Path(str(args.edits_json)).expanduser().resolve()
        self.preview_out = Path(str(args.preview_out)).expanduser().resolve()
        (self.preview_out / "masks").mkdir(parents=True, exist_ok=True)
        (self.preview_out / "vis").mkdir(parents=True, exist_ok=True)

        self.class_map = wbseg._parse_class_map(str(args.class_map))
        # Used for label generation/export behavior.
        self.include_aircraft = bool(int(args.include_aircraft_mask))
        # Used only for 2D editor visualization.
        self.show_aircraft_mask = bool(int(args.show_aircraft_mask))
        self.aircraft_class_id = int(np.clip(int(args.aircraft_class_id), 1, 255))
        self.alpha = float(np.clip(float(args.overlay_alpha), 0.0, 1.0))
        self.auto_update_yolo = bool(int(args.auto_update_yolo_labels))
        self.yolo_dataset_dir = Path(str(args.yolo_dataset_dir)).expanduser().resolve()
        self.yolo_split = str(args.yolo_split)
        self.min_contour_area = float(args.min_contour_area)
        self.contour_approx_eps = float(args.contour_approx_eps)
        self.yolo_class_id_to_idx: Dict[int, int] = {}
        self.yolo_idx_to_name: Dict[int, str] = {}

        try:
            import test_yolo_pose_from_h5_weights_to_pcd as pose_pcd
            import view_pcd_dir as pcd_view
        except Exception as e:
            raise RuntimeError("Failed importing project modules. Activate your project venv.") from e

        self.pose_pcd = pose_pcd
        self.pcd_view = pcd_view

        self.warning_state = wbseg._build_warning_state(args, pcd_view)
        self.h5_by_stem = wbseg._build_h5_index(self.source_root)
        self.edits_data = wbseg._load_edits_json(str(self.edits_json_path))

        if self.auto_update_yolo:
            self.yolo_class_id_to_idx, self.yolo_idx_to_name = wbseg._build_yolo_class_layout(
                class_map=self.class_map,
                include_aircraft_mask=self.include_aircraft,
                aircraft_class_id=self.aircraft_class_id,
            )
            for rel in (
                "images/train",
                "images/val",
                "images/test",
                "labels/train",
                "labels/val",
                "labels/test",
            ):
                (self.yolo_dataset_dir / rel).mkdir(parents=True, exist_ok=True)
            wbseg._write_yolo_dataset_yaml(
                yolo_root=self.yolo_dataset_dir,
                yaml_name=str(wbseg.DEFAULT_YOLO_YAML_NAME),
                yolo_idx_to_name=self.yolo_idx_to_name,
            )

        self.image_paths = wbseg._collect_images_from_dir(self.image_dir)
        total_images = int(len(self.image_paths))
        start_index = int(max(0, int(args.start_index)))
        if start_index > 0:
            if start_index >= total_images:
                raise RuntimeError(
                    f"No images after --start-index={start_index}. "
                    f"Found {total_images} images in {self.image_dir}. "
                    f"Use --start-index between 0 and {max(0, total_images - 1)}."
                )
            self.image_paths = self.image_paths[start_index:]
        if int(args.max_images) > 0:
            self.image_paths = self.image_paths[: int(args.max_images)]
        if not self.image_paths:
            raise RuntimeError(f"No images found in {self.image_dir}")
        self.bag_scene_counts: Dict[str, int] = {}
        for ip in self.image_paths:
            bag = _bag_prefix_from_unique_scene(str(ip.stem))
            self.bag_scene_counts[bag] = int(self.bag_scene_counts.get(bag, 0) + 1)

        self.idx = 0
        self.window = "Warning-Box Segmentation Editor (2D)"
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window, 1500, 900)

    def _make_mask_for_payload(self, payload: Dict[str, Any], *, include_aircraft: bool) -> np.ndarray:
        xyz_hw3 = np.asarray(payload["xyz_hw3"], dtype=np.float32)
        specs = payload["specs_cur"]
        mask_hw = wbseg._make_mask_from_specs(xyz_hw3=xyz_hw3, specs=specs, class_map=self.class_map)
        mask_hw = wbseg._apply_aircraft_mask(
            mask_hw=mask_hw,
            aircraft_mask_hw=payload["mask_air"] if bool(include_aircraft) else None,
            aircraft_class_id=self.aircraft_class_id,
            fill_background_only=True,
        )
        return np.asarray(mask_hw, dtype=np.uint8)

    def _save_yolo_scene_label(self, payload: Dict[str, Any]) -> None:
        if not self.auto_update_yolo:
            return
        mask_hw = self._make_mask_for_payload(payload, include_aircraft=self.include_aircraft)
        split = wbseg._resolve_yolo_split(Path(payload["image_path"]), self.yolo_split)
        unique_scene = str(payload["unique_scene"])
        lbl_dir = self.yolo_dataset_dir / "labels" / str(split)
        img_dir = self.yolo_dataset_dir / "images" / str(split)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        img_dir.mkdir(parents=True, exist_ok=True)

        yolo_lines, _inst = wbseg._mask_to_yolo_seg_lines(
            mask_hw=np.asarray(mask_hw, dtype=np.uint8),
            class_id_to_yolo=self.yolo_class_id_to_idx,
            min_contour_area=float(self.min_contour_area),
            contour_approx_eps=float(self.contour_approx_eps),
        )
        lbl_fp = lbl_dir / f"{unique_scene}.txt"
        lbl_fp.write_text(("\n".join(yolo_lines) + "\n") if yolo_lines else "", encoding="utf-8")

        # Keep image alongside label in target dataset.
        src_img = np.asarray(payload["image_bgr"], dtype=np.uint8)
        H, W = int(payload["H"]), int(payload["W"])
        if src_img.shape[0] != H or src_img.shape[1] != W:
            src_img = cv2.resize(src_img, (W, H), interpolation=cv2.INTER_LINEAR)
        img_fp = img_dir / f"{unique_scene}.png"
        cv2.imwrite(str(img_fp), src_img)
        print(f"[yolo] updated: {lbl_fp} instances={len(yolo_lines)} split={split}")

    def _scene_payload(self, ip: Path) -> Tuple[Optional[Dict[str, Any]], str]:
        bgr = cv2.imread(str(ip), cv2.IMREAD_COLOR)
        if bgr is None:
            return None, "unreadable_image"

        try:
            h5_stem, scene_name = wbseg._parse_unique_scene_stem(ip.stem)
        except Exception as e:
            return None, f"bad_scene_stem:{e}"
        unique_scene = f"{h5_stem}__{scene_name}"

        matches = self.h5_by_stem.get(str(h5_stem), [])
        if not matches:
            return None, f"h5_not_found:{h5_stem}"
        h5_path = matches[0]

        xyz_hw3, mask_air, H, W, xyz_reason = wbseg._load_scene_xyz(
            h5_path=h5_path,
            scene_name=str(scene_name),
            pose_mod=self.pose_pcd,
        )
        if xyz_hw3 is None:
            return None, f"xyz_missing:{xyz_reason}"

        specs_base, _checks, warn_reason = self.pcd_view._build_warning_specs_and_checks(
            unique_scene=unique_scene,
            kp_named=[],
            warning_state=self.warning_state,
            warning_box_scale=float(self.args.warning_box_scale),
        )
        if not specs_base:
            return None, f"warning_specs_missing:{warn_reason}"

        a380_front_override_status = "disabled"
        if bool(int(getattr(self.args, "a380_front_from_det_bbox", 1))):
            specs_base, a380_front_override_status = wbseg._apply_a380_front_spec_override_from_detection(
                specs=specs_base,
                unique_scene=unique_scene,
                image_path=ip,
                xyz_hw3=np.asarray(xyz_hw3, dtype=np.float32),
                det_dataset_root=str(getattr(self.args, "a380_det_dataset_root", wbseg.DEFAULT_A380_DET_DATASET_ROOT)),
                front_class_id=int(getattr(self.args, "a380_det_front_class_id", 3)),
                split_arg=str(getattr(self.args, "a380_det_split", "auto")),
                half_expand=float(getattr(self.args, "a380_det_half_expand", 1.0)),
                min_points=int(getattr(self.args, "a380_det_min_points", 30)),
            )

        specs_cur = wbseg._apply_scene_edits_to_specs(
            specs=specs_base,
            unique_scene=unique_scene,
            edits_data=self.edits_data,
        )
        return {
            "image_path": ip,
            "image_bgr": bgr,
            "h5_path": h5_path,
            "scene_name": scene_name,
            "unique_scene": unique_scene,
            "H": H,
            "W": W,
            "xyz_hw3": xyz_hw3,
            "mask_air": mask_air,
            "specs_base": _deepcopy_specs(specs_base),
            "specs_cur": _deepcopy_specs(specs_cur),
            "a380_front_override": str(a380_front_override_status),
        }, ""

    def _render_scene(self, payload: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mask_hw = self._make_mask_for_payload(payload, include_aircraft=self.show_aircraft_mask)

        src_img = np.asarray(payload["image_bgr"], dtype=np.uint8)
        H, W = int(payload["H"]), int(payload["W"])
        if src_img.shape[0] != H or src_img.shape[1] != W:
            src_img = cv2.resize(src_img, (W, H), interpolation=cv2.INTER_LINEAR)
        vis = wbseg._make_overlay(src_img, mask_hw, alpha=self.alpha)

        panel_w = 460
        canvas = np.zeros((H, W + panel_w, 3), dtype=np.uint8)
        canvas[:, :W] = vis
        panel = canvas[:, W:]
        panel[:, :] = (18, 18, 18)

        y = 28
        dy = 28
        cv2.putText(
            panel,
            "Warning-Box Segmentation Editor",
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        y += dy
        cv2.putText(
            panel,
            f"scene {self.idx+1}/{len(self.image_paths)}",
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        y += dy
        scene_txt = str(payload["unique_scene"])
        if len(scene_txt) > 66:
            scene_txt = scene_txt[:63] + "..."
        cv2.putText(
            panel,
            scene_txt,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )
        y += dy
        cv2.putText(
            panel,
            f"labeled_px={int(np.count_nonzero(mask_hw > 0))}",
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (180, 255, 180),
            1,
            cv2.LINE_AA,
        )
        y += dy
        cv2.putText(
            panel,
            f"edits_scenes={len(self.edits_data)}",
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (180, 255, 180),
            1,
            cv2.LINE_AA,
        )
        y += dy
        bag_prefix = _bag_prefix_from_unique_scene(str(payload["unique_scene"]))
        bag_count = int(self.bag_scene_counts.get(bag_prefix, 0))
        cv2.putText(
            panel,
            f"bag_scenes={bag_count} (g applies to this bag)",
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (180, 255, 220),
            1,
            cv2.LINE_AA,
        )
        y += int(1.0 * dy)
        ov_txt = f"a380_front_override={str(payload.get('a380_front_override', 'n/a'))}"
        if len(ov_txt) > 72:
            ov_txt = ov_txt[:69] + "..."
        cv2.putText(
            panel,
            ov_txt,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (180, 255, 200),
            1,
            cv2.LINE_AA,
        )
        y += dy
        controls = [
            "Controls:",
            "e = open 3D box editor",
            "n / b = next / previous image",
            "g (or B) = apply current edit to whole bag",
            "s = save edits JSON",
            "m = save preview mask+overlay",
            "r = reset scene edit",
            "q = quit",
        ]
        for i, t in enumerate(controls):
            color = (230, 230, 230) if i == 0 else (190, 230, 190)
            cv2.putText(
                panel,
                t,
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                (0.56 if i == 0 else 0.50),
                color,
                1,
                cv2.LINE_AA,
            )
            y += dy

        return canvas, mask_hw, vis

    def _persist_scene_edit(self, payload: Dict[str, Any]) -> None:
        unique_scene = str(payload["unique_scene"])
        base_specs = payload["specs_base"]
        cur_specs = payload["specs_cur"]

        if _specs_equal(base_specs, cur_specs):
            if unique_scene in self.edits_data:
                del self.edits_data[unique_scene]
                print(f"[edit] removed scene edit (back to default): {unique_scene}")
            return

        rec: Dict[str, Any] = {}
        for k, s in cur_specs.items():
            rec[str(k)] = _spec_to_jsonable(s)
        self.edits_data[unique_scene] = rec
        print(f"[edit] scene updated: {unique_scene}")

    def _save_preview(self, payload: Dict[str, Any], mask_hw: np.ndarray, vis: np.ndarray) -> None:
        scene = str(payload["unique_scene"])
        mask_fp = self.preview_out / "masks" / f"{scene}.png"
        vis_fp = self.preview_out / "vis" / f"{scene}.png"
        cv2.imwrite(str(mask_fp), np.asarray(mask_hw, dtype=np.uint8))
        cv2.imwrite(str(vis_fp), np.asarray(vis, dtype=np.uint8))
        print(f"[preview] saved: {mask_fp} | {vis_fp}")

    def _apply_current_edit_to_bag(self, payload_src: Dict[str, Any]) -> None:
        src_scene = str(payload_src["unique_scene"])
        bag_prefix = _bag_prefix_from_unique_scene(src_scene)
        delta_map = _compute_delta_map(
            base_specs=payload_src["specs_base"],
            cur_specs=payload_src["specs_cur"],
        )
        if not delta_map:
            print(f"[bag-apply] no specs to apply for source scene: {src_scene}")
            return

        bag_images = [
            ip for ip in self.image_paths
            if str(ip.stem).startswith(f"{bag_prefix}__")
        ]
        if not bag_images:
            print(f"[bag-apply] no scenes found for bag: {bag_prefix}")
            return

        print(f"[bag-apply] start source={src_scene} bag={bag_prefix}")
        processed = 0
        updated = 0
        removed = 0
        skipped = 0
        yolo_updates = 0

        for ip in bag_images:
            payload_t, reason = self._scene_payload(ip)
            if payload_t is None:
                skipped += 1
                print(f"[bag-apply] skip {ip.name}: {reason}")
                continue
            scene_t = str(payload_t["unique_scene"])
            base_t = _deepcopy_specs(payload_t["specs_base"])
            if scene_t == src_scene:
                new_specs = _deepcopy_specs(payload_src["specs_cur"])
            else:
                new_specs = _apply_delta_map_to_specs(base_t, delta_map)

            if _specs_equal(base_t, new_specs):
                if scene_t in self.edits_data:
                    del self.edits_data[scene_t]
                    removed += 1
            else:
                rec: Dict[str, Any] = {}
                for k, s in new_specs.items():
                    rec[str(k)] = _spec_to_jsonable(s)
                self.edits_data[scene_t] = rec
                updated += 1

            if self.auto_update_yolo:
                payload_t["specs_cur"] = _deepcopy_specs(new_specs)
                self._save_yolo_scene_label(payload_t)
                yolo_updates += 1
            processed += 1

        _save_edits_json(self.edits_json_path, self.edits_data)
        print(
            "[bag-apply] "
            f"bag={bag_prefix} scenes={len(bag_images)} processed={processed} "
            f"updated={updated} removed={removed} skipped={skipped} "
            f"yolo_updated={yolo_updates}"
        )

    def run(self) -> None:
        print(f"[session] images={len(self.image_paths)}")
        print(f"[session] source_h5_root={self.source_root}")
        print(f"[session] edits_json={self.edits_json_path}")
        print(
            f"[session] show_aircraft_mask={int(self.show_aircraft_mask)} "
            f"include_aircraft_for_labels={int(self.include_aircraft)}"
        )
        if self.auto_update_yolo:
            print(f"[session] yolo auto-update=ON dataset={self.yolo_dataset_dir}")
        else:
            print("[session] yolo auto-update=OFF (manual regenerate needed)")

        while True:
            ip = self.image_paths[self.idx]
            payload, reason = self._scene_payload(ip)
            if payload is None:
                canvas = np.zeros((400, 1200, 3), dtype=np.uint8)
                msg = f"[{self.idx+1}/{len(self.image_paths)}] {ip.name} -> {reason}"
                cv2.putText(canvas, msg, (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)
                cv2.putText(canvas, "n=next, b=prev, q=quit", (12, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 255, 180), 1, cv2.LINE_AA)
                cv2.imshow(self.window, canvas)
                k = cv2.waitKey(0) & 0xFF
                if k == ord("q"):
                    _save_edits_json(self.edits_json_path, self.edits_data)
                    break
                if k == ord("n"):
                    self.idx = min(len(self.image_paths) - 1, self.idx + 1)
                elif k == ord("b"):
                    self.idx = max(0, self.idx - 1)
                elif k == ord("s"):
                    _save_edits_json(self.edits_json_path, self.edits_data)
                continue

            top, mask_hw, vis = self._render_scene(payload)
            cv2.imshow(self.window, top)
            k = cv2.waitKey(0) & 0xFF

            if k == ord("q"):
                self._persist_scene_edit(payload)
                _save_edits_json(self.edits_json_path, self.edits_data)
                self._save_yolo_scene_label(payload)
                break
            if k == ord("s"):
                self._persist_scene_edit(payload)
                _save_edits_json(self.edits_json_path, self.edits_data)
                self._save_yolo_scene_label(payload)
                continue
            if k == ord("m"):
                self._persist_scene_edit(payload)
                self._save_preview(payload, mask_hw, vis)
                self._save_yolo_scene_label(payload)
                continue
            if k in (ord("g"), ord("G"), ord("B")):
                self._persist_scene_edit(payload)
                self._apply_current_edit_to_bag(payload)
                self._save_yolo_scene_label(payload)
                continue
            if k == ord("r"):
                scene = str(payload["unique_scene"])
                payload["specs_cur"] = _deepcopy_specs(payload["specs_base"])
                if scene in self.edits_data:
                    del self.edits_data[scene]
                    print(f"[edit] reset scene edit: {scene}")
                _save_edits_json(self.edits_json_path, self.edits_data)
                self._save_yolo_scene_label(payload)
                continue
            if k == ord("e"):
                try:
                    specs_edited, changed = _edit_specs_in_3d(
                        xyz_hw3=np.asarray(payload["xyz_hw3"], dtype=np.float32),
                        specs=payload["specs_cur"],
                        scene_name=str(payload["unique_scene"]),
                        point_limit=int(self.args.point_limit),
                        trans_step=float(self.args.trans_step),
                        rot_step_deg=float(self.args.rot_step_deg),
                        scale_step=float(self.args.scale_step),
                        pcd_view=self.pcd_view,
                    )
                    if changed:
                        payload["specs_cur"] = _deepcopy_specs(specs_edited)
                        self._persist_scene_edit(payload)
                        _save_edits_json(self.edits_json_path, self.edits_data)
                        self._save_yolo_scene_label(payload)
                except Exception as e:
                    print(f"[3d] editor failed: {type(e).__name__}: {e}")
                continue
            if k == ord("n"):
                self._persist_scene_edit(payload)
                self._save_yolo_scene_label(payload)
                self.idx = min(len(self.image_paths) - 1, self.idx + 1)
                continue
            if k == ord("b"):
                self._persist_scene_edit(payload)
                self._save_yolo_scene_label(payload)
                self.idx = max(0, self.idx - 1)
                continue

        cv2.destroyAllWindows()


def main() -> None:
    args = _parse_args()
    sess = Session(args)
    sess.run()


if __name__ == "__main__":
    main()
