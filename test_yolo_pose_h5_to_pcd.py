#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run a YOLO-Pose .pt model on images built from H5 range images and
save pointclouds cropped to the predicted bbox as .pcd (ASCII).

This script reuses the image-building logic from projection_helpers.py.
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import h5py
import cv2

from config_dataset import SOURCE, APPLY_MEDIAN_FILTER, MEDIAN_KSIZE
from io_helpers import list_h5_paths, open_h5_any
from projection_helpers import build_rgb_from_cols


def write_pcd_xyz(path: Path, points_xyz: np.ndarray) -> None:
    """Write Nx3 pointcloud to ASCII .pcd."""
    pts = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
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
        for x, y, z in pts:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def _pick_best_bbox(result, target_class: int = 0) -> Optional[Tuple[int, int, int, int]]:
    """Return (x1,y1,x2,y2) for the highest-confidence bbox of target_class."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None

    cls = boxes.cls.detach().cpu().numpy().astype(int)
    conf = boxes.conf.detach().cpu().numpy()
    xyxy = boxes.xyxy.detach().cpu().numpy()

    best_i = None
    best_c = -1.0
    for i in range(len(xyxy)):
        if cls[i] != target_class:
            continue
        if conf[i] > best_c:
            best_c = float(conf[i])
            best_i = i

    if best_i is None:
        return None

    x1, y1, x2, y2 = xyxy[best_i].tolist()
    return int(x1), int(y1), int(x2), int(y2)


def _build_rgb_and_xyz(flat: np.ndarray, cols: List[str], H: int, W: int):
    """Build RGB image and XYZ grid (H,W,3)."""
    if not {"x", "y", "z"}.issubset(set(cols)):
        return None, None
    ix, iy, iz = cols.index("x"), cols.index("y"), cols.index("z")

    rgb = build_rgb_from_cols(flat, cols, H, W)
    if rgb is None:
        return None, None

    if APPLY_MEDIAN_FILTER:
        rgb = cv2.medianBlur(rgb.astype(np.uint8), MEDIAN_KSIZE)

    xyz = np.stack([flat[:, ix], flat[:, iy], flat[:, iz]], axis=1).astype(np.float32)
    return rgb, xyz.reshape(H, W, 3)


def run(
    source: str,
    weights: str,
    out_dir: str,
    max_h5_files: Optional[int] = None,
    imgsz: int = 1024,
    conf: float = 0.25,
    device: str = "0",
) -> None:
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError(
            "ultralytics is required. Install with `pip install ultralytics`."
        ) from e

    model = YOLO(weights)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    print("[list] Searching for .h5 files…")
    h5_paths = list_h5_paths(source)
    if not h5_paths:
        raise RuntimeError(f"No .h5 files found under: {source}")

    if max_h5_files is not None and max_h5_files > 0 and len(h5_paths) > max_h5_files:
        h5_paths = h5_paths[:max_h5_files]
        print(f"[list] Using only first {len(h5_paths)} HDF5 files (max_h5_files={max_h5_files})")

    total_scenes = 0
    total_saved = 0

    for i, h5p in enumerate(h5_paths, 1):
        print(f"[{i}/{len(h5_paths)}] {Path(h5p).name}")
        try:
            with open_h5_any(h5p) as f:
                H = int(f.attrs["height"])
                W = int(f.attrs["width"])

                for scene_name, grp in f.items():
                    if not isinstance(grp, h5py.Group) or "points" not in grp:
                        continue

                    total_scenes += 1
                    file_stem = Path(h5p).stem
                    unique_scene = f"{file_stem}__{scene_name}"

                    ds = grp["points"]
                    flat = ds[()]  # (H*W, C)

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

                    rgb, xyz_hw3 = _build_rgb_and_xyz(flat, cols, H, W)
                    if rgb is None or xyz_hw3 is None:
                        print(f"  [SKIP] Missing required cols for {unique_scene}")
                        continue

                    # YOLO inference
                    results = model.predict(rgb, imgsz=imgsz, conf=conf, device=device, verbose=False)
                    if not results:
                        print(f"  [SKIP] No results for {unique_scene}")
                        continue

                    bb = _pick_best_bbox(results[0], target_class=0)
                    if bb is None:
                        print(f"  [SKIP] No aircraft bbox for {unique_scene}")
                        continue

                    x1, y1, x2, y2 = bb
                    x1 = int(np.clip(x1, 0, W - 1))
                    x2 = int(np.clip(x2, 0, W - 1))
                    y1 = int(np.clip(y1, 0, H - 1))
                    y2 = int(np.clip(y2, 0, H - 1))
                    if x2 < x1 or y2 < y1:
                        print(f"  [SKIP] Invalid bbox for {unique_scene}")
                        continue

                    # crop points to bbox
                    pts = xyz_hw3[y1 : y2 + 1, x1 : x2 + 1, :].reshape(-1, 3)
                    finite = np.all(np.isfinite(pts), axis=1)
                    pts = pts[finite]
                    if pts.size == 0:
                        print(f"  [SKIP] Empty pointcloud for {unique_scene}")
                        continue

                    out_path = out_root / f"{unique_scene}.pcd"
                    write_pcd_xyz(out_path, pts)
                    total_saved += 1

        except Exception as e:
            print(f"[ERROR] {Path(h5p).name}: {e}")

    print(f"\n[summary] scenes processed: {total_scenes}")
    print(f"[summary] pcd saved: {total_saved}")
    print("  -> output:", out_root.resolve())


def parse_args():
    p = argparse.ArgumentParser(
        description="Run YOLO-Pose .pt on H5 range images and save bbox-cropped pointclouds as .pcd"
    )
    p.add_argument("--source", type=str, default=SOURCE, help="Input H5 directory or gs:// path")
    p.add_argument("--weights", type=str, required=True, help="YOLO .pt weights")
    p.add_argument("--out", type=str, default="./pcd_from_yolo", help="Output directory for .pcd files")
    p.add_argument("--max-h5", type=int, default=None, help="Limit number of H5 files")
    p.add_argument("--imgsz", type=int, default=1024, help="YOLO inference image size")
    p.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    p.add_argument("--device", type=str, default="0", help="YOLO device (e.g. 0, 0,1, or cpu)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        source=args.source,
        weights=args.weights,
        out_dir=args.out,
        max_h5_files=args.max_h5,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
    )
