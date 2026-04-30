#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple viewer for debug point-cloud keypoint PLY files.

Examples:
  python view_debug_pointcloud_kps.py
  python view_debug_pointcloud_kps.py --root ./aircraft_pose_with_normalising_applied_grayscale/debug_pointcloud_kps --split train --max 20
  python view_debug_pointcloud_kps.py --file ./aircraft_pose_with_normalising_applied_grayscale/debug_pointcloud_kps/train/scene_xxx.ply
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np


def _collect_ply_files(root: Path, split: str, pattern: str) -> List[Path]:
    if split.lower() == "all":
        files = sorted(root.rglob("*.ply"))
    else:
        files = sorted((root / split).rglob("*.ply"))
    if pattern:
        p = pattern.lower()
        files = [f for f in files if p in f.name.lower()]
    return files


def _parse_legend_points(legend_path: Path) -> Tuple[List[str], Dict[str, np.ndarray]]:
    lines: List[str] = []
    pts: Dict[str, np.ndarray] = {}
    if not legend_path.exists():
        return lines, pts

    lines = legend_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    pat = re.compile(
        r"^\s*([^:]+)\s*:\s*xyz=\(\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*\)"
    )
    for ln in lines:
        m = pat.match(ln)
        if not m:
            continue
        name = str(m.group(1)).strip()
        try:
            x = float(m.group(2))
            y = float(m.group(3))
            z = float(m.group(4))
            pts[name] = np.array([x, y, z], dtype=np.float64)
        except Exception:
            continue
    return lines, pts


def _looks_like_warning_anchor(name: str) -> bool:
    n = name.lower().strip()
    if "warning" in n:
        return True
    if "box_center" in n:
        return True
    return n in {"front_landing_gear", "engine_left", "engine_right"}


def _warning_color_rgb01(name: str) -> np.ndarray:
    n = name.lower()
    if "engine_left" in n:
        return np.array([1.0, 0.62, 0.05], dtype=np.float64)
    if "engine_right" in n:
        return np.array([0.05, 0.85, 1.0], dtype=np.float64)
    if "landing_gear" in n or "front" in n:
        return np.array([1.0, 0.2, 0.9], dtype=np.float64)
    return np.array([1.0, 0.35, 0.1], dtype=np.float64)


def _warning_size_multiplier(name: str) -> float:
    n = name.lower()
    if "landing_gear" in n or "front" in n:
        return 0.75
    if "engine" in n:
        return 1.0
    return 0.85


def _box_corners_and_edges(center: np.ndarray, half_xy: float, half_z: float) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    cx, cy, cz = [float(v) for v in center]
    corners = np.array(
        [
            [cx - half_xy, cy - half_xy, cz - half_z],
            [cx + half_xy, cy - half_xy, cz - half_z],
            [cx + half_xy, cy + half_xy, cz - half_z],
            [cx - half_xy, cy + half_xy, cz - half_z],
            [cx - half_xy, cy - half_xy, cz + half_z],
            [cx + half_xy, cy - half_xy, cz + half_z],
            [cx + half_xy, cy + half_xy, cz + half_z],
            [cx - half_xy, cy + half_xy, cz + half_z],
        ],
        dtype=np.float64,
    )
    edges: List[Tuple[int, int]] = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    return corners, edges


def _auto_warning_box_size(xyz: np.ndarray) -> float:
    if xyz.size == 0:
        return 1.2
    ext = np.ptp(xyz, axis=0)
    m = float(np.max(ext))
    if not np.isfinite(m) or m <= 0:
        return 1.2
    return float(np.clip(0.06 * m, 0.5, 5.0))


def _show_one(
    ply_path: Path,
    *,
    show_warning_boxes: bool,
    warning_box_size: float,
    warning_box_z_scale: float,
    show_axes: bool,
) -> bool:
    try:
        import open3d as o3d
        use_open3d = True
    except Exception as e:
        print(f"[WARN] open3d not available ({e}). Falling back to matplotlib.")
        use_open3d = False

    if not ply_path.exists():
        print(f"[ERROR] File not found: {ply_path}")
        return False

    legend = ply_path.with_suffix(".txt")
    legend_lines, legend_points = _parse_legend_points(legend)
    warning_pts = {k: v for k, v in legend_points.items() if _looks_like_warning_anchor(k)}

    print(f"\nViewing: {ply_path}")
    if legend_lines:
        print("Legend:")
        for ln in legend_lines[:15]:
            print(f"  {ln}")
        if len(legend_lines) > 15:
            print(f"  ... ({len(legend_lines) - 15} more)")
    if show_warning_boxes:
        if warning_pts:
            print(f"[INFO] Warning-box anchors found: {len(warning_pts)}")
            for nm, p in sorted(warning_pts.items()):
                print(f"  - {nm}: ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})")
        else:
            print("[INFO] No warning-box anchors found in legend.")

    if use_open3d:
        pcd = o3d.io.read_point_cloud(str(ply_path))
        if pcd.is_empty():
            print(f"[WARN] Empty point cloud: {ply_path}")
            return True
        geoms: List[object] = [pcd]
        if show_warning_boxes and warning_pts:
            cloud_xyz = np.asarray(pcd.points)
            base_size = float(warning_box_size) if float(warning_box_size) > 0 else _auto_warning_box_size(cloud_xyz)
            for nm, ctr in sorted(warning_pts.items()):
                size_full = base_size * _warning_size_multiplier(nm)
                half_xy = max(0.05, 0.5 * float(size_full))
                half_z = max(0.05, 0.5 * float(size_full) * float(warning_box_z_scale))
                corners, edges = _box_corners_and_edges(ctr, half_xy, half_z)
                ls = o3d.geometry.LineSet()
                ls.points = o3d.utility.Vector3dVector(corners)
                ls.lines = o3d.utility.Vector2iVector(np.asarray(edges, dtype=np.int32))
                col = _warning_color_rgb01(nm).reshape(1, 3)
                ls.colors = o3d.utility.Vector3dVector(np.tile(col, (len(edges), 1)))
                geoms.append(ls)
        o3d.visualization.draw_geometries(
            geoms,
            window_name=f"KP Debug: {ply_path.name}",
            width=1400,
            height=900,
        )
        return True

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[ERROR] matplotlib is not available: {e}")
        return False

    try:
        with ply_path.open("r", encoding="utf-8", errors="ignore") as f:
            line = f.readline()
            if "ply" not in line.lower():
                print(f"[ERROR] Not a PLY file: {ply_path}")
                return False
            n_vertices = None
            header_lines = 1
            while True:
                line = f.readline()
                if not line:
                    print(f"[ERROR] Invalid PLY header: {ply_path}")
                    return False
                header_lines += 1
                l = line.strip().lower()
                if l.startswith("element vertex"):
                    parts = l.split()
                    if len(parts) >= 3:
                        n_vertices = int(parts[2])
                if l == "end_header":
                    break
        if not n_vertices or n_vertices <= 0:
            print(f"[WARN] No vertices found: {ply_path}")
            return True

        arr = np.loadtxt(str(ply_path), dtype=np.float64, skiprows=header_lines, max_rows=n_vertices)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] < 3:
            print(f"[ERROR] PLY does not contain xyz columns: {ply_path}")
            return False

        xyz = arr[:, :3]
        if arr.shape[1] >= 6:
            rgb = np.clip(arr[:, 3:6], 0, 255) / 255.0
        else:
            rgb = np.full((xyz.shape[0], 3), 0.65, dtype=np.float64)

        base_size = float(warning_box_size) if float(warning_box_size) > 0 else _auto_warning_box_size(xyz)

        # Keep plotting responsive for very dense files.
        if xyz.shape[0] > 120000:
            idx = np.random.choice(xyz.shape[0], size=120000, replace=False)
            xyz = xyz[idx]
            rgb = rgb[idx]

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=rgb, s=0.7, depthshade=False)
        if show_axes:
            ax.set_title(f"KP Debug: {ply_path.name}")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
        else:
            # Raw point-cloud style view (no graph frame/axes).
            ax.set_axis_off()
            ax.grid(False)
        if show_warning_boxes and warning_pts:
            for nm, ctr in sorted(warning_pts.items()):
                size_full = base_size * _warning_size_multiplier(nm)
                half_xy = max(0.05, 0.5 * float(size_full))
                half_z = max(0.05, 0.5 * float(size_full) * float(warning_box_z_scale))
                corners, edges = _box_corners_and_edges(ctr, half_xy, half_z)
                col = _warning_color_rgb01(nm)
                for i0, i1 in edges:
                    seg = corners[[i0, i1]]
                    ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=col, linewidth=1.3)
                if show_axes:
                    ax.text(float(ctr[0]), float(ctr[1]), float(ctr[2]), nm, color=col, fontsize=7)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"[ERROR] Failed to open with matplotlib: {e}")
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="View debug point-cloud keypoint PLY files.")
    parser.add_argument(
        "--root",
        type=str,
        default="./aircraft_pose_with_normalising_applied_grayscale/debug_pointcloud_kps",
        help="Root folder containing split subfolders with .ply files.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["all", "train", "val", "test"],
        help="Which split folder to browse.",
    )
    parser.add_argument("--pattern", type=str, default="", help="Filter by substring in filename.")
    parser.add_argument("--max", type=int, default=0, help="Max files to view (0 = all).")
    parser.add_argument("--start", type=int, default=0, help="Start index for browsing.")
    parser.add_argument("--list", action="store_true", help="Only list files; do not open viewer.")
    parser.add_argument("--file", type=str, default="", help="Open one specific .ply file.")
    parser.add_argument(
        "--no-warning-boxes",
        action="store_true",
        help="Disable warning-box placement overlay parsed from the scene .txt legend.",
    )
    parser.add_argument(
        "--warning-box-size",
        type=float,
        default=0.0,
        help="Warning-box full size in meters (0 = auto from cloud extent).",
    )
    parser.add_argument(
        "--warning-box-z-scale",
        type=float,
        default=0.65,
        help="Relative Z size scale for warning boxes.",
    )
    parser.add_argument(
        "--show-axes",
        action="store_true",
        help="Show matplotlib graph axes/title/labels (default is raw point-cloud view without axes).",
    )
    args = parser.parse_args()

    show_warning_boxes = not bool(args.no_warning_boxes)

    if args.file:
        _show_one(
            Path(args.file).expanduser().resolve(),
            show_warning_boxes=show_warning_boxes,
            warning_box_size=float(args.warning_box_size),
            warning_box_z_scale=float(args.warning_box_z_scale),
            show_axes=bool(args.show_axes),
        )
        return

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        print(f"[ERROR] Root not found: {root}")
        return

    files = _collect_ply_files(root=root, split=args.split, pattern=args.pattern)
    if not files:
        print(f"[INFO] No .ply files found under: {root} (split={args.split}, pattern='{args.pattern}')")
        return

    start = max(0, int(args.start))
    files = files[start:]
    if args.max and args.max > 0:
        files = files[: args.max]

    print(f"[INFO] Found {len(files)} file(s).")
    for i, p in enumerate(files, 1):
        print(f"{i:04d}: {p}")
    if args.list:
        return

    for i, p in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] Close the window to continue...")
        ok = _show_one(
            p,
            show_warning_boxes=show_warning_boxes,
            warning_box_size=float(args.warning_box_size),
            warning_box_z_scale=float(args.warning_box_z_scale),
            show_axes=bool(args.show_axes),
        )
        if not ok:
            break


if __name__ == "__main__":
    main()
