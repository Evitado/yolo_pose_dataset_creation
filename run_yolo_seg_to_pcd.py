#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run YOLO-seg on image(s) and backproject predicted masks to point clouds.

For each image named as:
  <h5_stem>__<scene_name>.<ext>
the script:
1) loads matching H5 scene XYZ
2) runs YOLO segmentation weights on the image
3) unions masks per class
4) saves class-wise and combined point clouds as ASCII .pcd
5) writes a CSV summary
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


DEFAULT_IMAGE_PATH = "/home/femi/yolo_pose_dataset_creation/aircraft_pose_with_normalising_applied_multifield_only_3_2/images/test"
DEFAULT_SOURCE_H5_ROOT = "/home/femi/Benchmarking_framework/Data/warning_b_test_h5"
DEFAULT_WEIGHTS = "/home/femi/Benchmarking_framework/runs/segment/train-4/weights/best.pt"
DEFAULT_OUT_DIR = "/home/femi/yolo_pose_dataset_creation/pcd_from_yolo_seg"


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run YOLO segmentation and backproject predicted masks to PCD."
    )
    ap.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS, help="YOLO segmentation weights (.pt)")
    ap.add_argument("--image-path", type=str, default=DEFAULT_IMAGE_PATH, help="Input image file or directory")
    ap.add_argument("--image-dir", type=str, default="", help="Alternative image directory")
    ap.add_argument("--source", type=str, default=DEFAULT_SOURCE_H5_ROOT, help="Root containing source H5 files")
    ap.add_argument("--out", type=str, default=DEFAULT_OUT_DIR, help="Output directory")
    ap.add_argument("--conf", type=float, default=0.10, help="YOLO confidence threshold")
    ap.add_argument("--iou", type=float, default=0.7, help="YOLO NMS IoU threshold")
    ap.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    ap.add_argument("--max-det", type=int, default=300, help="YOLO max detections")
    ap.add_argument("--device", type=str, default="", help="YOLO device (e.g. 0, cpu)")
    ap.add_argument("--class-ids", type=str, default="", help="Optional class-id filter, e.g. '0,1,2'")
    ap.add_argument("--max-images", type=int, default=0, help="Maximum images to process (0=all)")
    ap.add_argument("--min-points", type=int, default=5, help="Min points required to save a PCD file")
    ap.add_argument(
        "--mask-open-ksize",
        type=int,
        default=0,
        help="Mask morphological opening kernel size (<=1 disables).",
    )
    ap.add_argument(
        "--mask-close-ksize",
        type=int,
        default=3,
        help="Mask morphological closing kernel size (<=1 disables).",
    )
    ap.add_argument(
        "--mask-min-area",
        type=int,
        default=0,
        help="Remove connected components smaller than this pixel area (<=0 disables).",
    )
    ap.add_argument(
        "--knn-filter",
        type=int,
        default=1,
        choices=[0, 1],
        help="Apply 3D KNN/statistical outlier filtering before saving PCD.",
    )
    ap.add_argument(
        "--knn-k",
        type=int,
        default=20,
        help="K neighbors used in KNN outlier filtering.",
    )
    ap.add_argument(
        "--knn-std-ratio",
        type=float,
        default=2.5,
        help="Outlier threshold = mean(nn_dist) + std_ratio*std(nn_dist).",
    )
    ap.add_argument(
        "--voxel-size",
        type=float,
        default=0.0,
        help="Voxel downsample size in world units before saving PCD (<=0 disables).",
    )
    ap.add_argument(
        "--expand-radius-m",
        type=float,
        default=0.3,
        help="3D expansion radius around each class segmentation in meters (<=0 disables).",
    )
    ap.add_argument("--save-vis", type=int, default=1, choices=[0, 1], help="Save prediction overlay images")
    ap.add_argument(
        "--show-vis",
        type=int,
        default=0,
        choices=[0, 1],
        help="Show live per-scene overlay window while processing.",
    )
    ap.add_argument(
        "--show-vis-ms",
        type=int,
        default=0,
        help="If >0, auto-advance each scene after this many ms. If 0, wait for key.",
    )
    ap.add_argument(
        "--show-vis-window",
        type=str,
        default="YOLO Seg To PCD",
        help="Window title for --show-vis.",
    )
    ap.add_argument(
        "--show-pcd",
        type=int,
        default=1,
        choices=[0, 1],
        help="Show Open3D pointcloud window for each processed scene.",
    )
    ap.add_argument(
        "--show-pcd-max-points",
        type=int,
        default=120000,
        help="Max points rendered per cloud in --show-pcd mode.",
    )
    return ap.parse_args()


def _to_int_set(raw: str) -> Optional[set[int]]:
    s = str(raw or "").strip()
    if not s:
        return None
    out: set[int] = set()
    for tok in s.split(","):
        t = str(tok).strip()
        if not t:
            continue
        out.add(int(t))
    return out if out else None


def _sanitize_name(name: str) -> str:
    s = str(name or "").strip()
    s = re.sub(r"[^a-zA-Z0-9_.-]+", "_", s)
    return s or "class"


def _as_class_name_map(names_obj: Any) -> Dict[int, str]:
    if isinstance(names_obj, dict):
        out: Dict[int, str] = {}
        for k, v in names_obj.items():
            try:
                out[int(k)] = str(v)
            except Exception:
                continue
        return out
    if isinstance(names_obj, (list, tuple)):
        return {int(i): str(v) for i, v in enumerate(names_obj)}
    return {}


def _write_pcd_xyz(path: Path, points_xyz: np.ndarray) -> None:
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(header + "\n")
        for x, y, z in pts:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def _collect_images(image_path_raw: str, image_dir_raw: str, wbseg: Any) -> List[Path]:
    image_paths: List[Path] = []
    ipath = str(image_path_raw or "").strip()
    idir = str(image_dir_raw or "").strip()

    if ipath and idir:
        if ipath == str(wbseg.DEFAULT_IMAGE_PATH):
            ipath = ""
        else:
            raise RuntimeError("Use only one of --image-path or --image-dir.")

    if idir:
        image_paths = wbseg._collect_images_from_dir(Path(idir).expanduser().resolve())
    elif ipath:
        p = Path(ipath).expanduser().resolve()
        if p.exists() and p.is_dir():
            image_paths = wbseg._collect_images_from_dir(p)
        elif p.exists() and p.is_file():
            image_paths = [p]
        else:
            raise RuntimeError(f"Input path not found: {p}")
    else:
        raise RuntimeError("Please provide --image-path or --image-dir.")
    return image_paths


def _resize_masks_to_hw(masks_nhw: np.ndarray, H: int, W: int) -> np.ndarray:
    arr = np.asarray(masks_nhw, dtype=np.float32)
    if arr.ndim != 3:
        return np.zeros((0, H, W), dtype=np.float32)
    n, h, w = arr.shape
    if h == H and w == W:
        return arr
    out = np.zeros((n, H, W), dtype=np.float32)
    for i in range(n):
        out[i] = cv2.resize(arr[i], (W, H), interpolation=cv2.INTER_LINEAR)
    return out


def _to_odd_ksize(v: int) -> int:
    k = int(v)
    if k <= 1:
        return 0
    if (k % 2) == 0:
        k += 1
    return k


def _clean_binary_mask(
    mask_hw: np.ndarray,
    *,
    open_ksize: int,
    close_ksize: int,
    min_area: int,
) -> np.ndarray:
    m = np.asarray(mask_hw, dtype=bool)
    if m.size == 0:
        return m
    u = (m.astype(np.uint8) * 255)

    k_open = _to_odd_ksize(int(open_ksize))
    if k_open > 1:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
        u = cv2.morphologyEx(u, cv2.MORPH_OPEN, ker)

    k_close = _to_odd_ksize(int(close_ksize))
    if k_close > 1:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
        u = cv2.morphologyEx(u, cv2.MORPH_CLOSE, ker)

    min_area_i = int(min_area)
    if min_area_i > 0:
        n_lab, lab, stats, _ = cv2.connectedComponentsWithStats((u > 0).astype(np.uint8), connectivity=8)
        keep = np.zeros_like(u, dtype=np.uint8)
        for li in range(1, int(n_lab)):
            if int(stats[li, cv2.CC_STAT_AREA]) >= min_area_i:
                keep[lab == li] = 255
        u = keep
    return u > 0


def _apply_mask_cleanup(
    class_masks: Dict[int, np.ndarray],
    *,
    open_ksize: int,
    close_ksize: int,
    min_area: int,
) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    for cid, m in class_masks.items():
        out[int(cid)] = _clean_binary_mask(
            m,
            open_ksize=int(open_ksize),
            close_ksize=int(close_ksize),
            min_area=int(min_area),
        )
    return out


def _voxel_downsample_grid(points_xyz: np.ndarray, voxel_size: float) -> np.ndarray:
    pts = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    vs = float(voxel_size)
    if pts.shape[0] <= 1 or vs <= 0.0:
        return pts
    keys = np.floor(pts / vs).astype(np.int64)
    _, keep_idx = np.unique(keys, axis=0, return_index=True)
    keep_idx = np.sort(keep_idx.astype(np.int64))
    return pts[keep_idx]


def _knn_statistical_filter(
    points_xyz: np.ndarray,
    *,
    k: int,
    std_ratio: float,
    ckdtree_cls: Optional[Any],
) -> np.ndarray:
    pts = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    if ckdtree_cls is None:
        return pts
    if pts.shape[0] <= 2:
        return pts

    k_eff = int(max(2, int(k)))
    qk = int(min(int(pts.shape[0]), k_eff + 1))
    if qk <= 1:
        return pts

    tree = ckdtree_cls(pts.astype(np.float64))
    try:
        dists, _ = tree.query(pts.astype(np.float64), k=qk, workers=-1)
    except TypeError:
        dists, _ = tree.query(pts.astype(np.float64), k=qk)

    d = np.asarray(dists, dtype=np.float64)
    if d.ndim != 2 or d.shape[1] <= 1:
        return pts
    mean_nn = np.mean(d[:, 1:], axis=1)
    mu = float(np.mean(mean_nn))
    sigma = float(np.std(mean_nn))
    if not np.isfinite(mu) or not np.isfinite(sigma):
        return pts
    if sigma <= 1e-12:
        return pts

    thr = float(mu + float(std_ratio) * sigma)
    keep = mean_nn <= thr
    if int(np.count_nonzero(keep)) <= 0:
        return pts
    return pts[keep]


def _expand_seeds_to_scene_points(
    seed_points_xyz: np.ndarray,
    scene_points_xyz: np.ndarray,
    *,
    radius_m: float,
    ckdtree_cls: Optional[Any],
) -> Tuple[np.ndarray, np.ndarray]:
    seeds = np.asarray(seed_points_xyz, dtype=np.float32).reshape(-1, 3)
    scene = np.asarray(scene_points_xyz, dtype=np.float32).reshape(-1, 3)
    if scene.shape[0] <= 0:
        return np.zeros((0,), dtype=bool), np.zeros((0, 3), dtype=np.float32)
    if seeds.shape[0] <= 0:
        return np.zeros((scene.shape[0],), dtype=bool), np.zeros((0, 3), dtype=np.float32)
    if ckdtree_cls is None or float(radius_m) <= 0.0:
        return np.zeros((scene.shape[0],), dtype=bool), seeds

    tree = ckdtree_cls(seeds.astype(np.float64))
    try:
        dists, _ = tree.query(scene.astype(np.float64), k=1, workers=-1)
    except TypeError:
        dists, _ = tree.query(scene.astype(np.float64), k=1)
    keep = np.asarray(dists, dtype=np.float64) <= float(radius_m)
    return keep, scene[keep]


def _build_class_union_masks(
    *,
    result: Any,
    H: int,
    W: int,
    conf_thr: float,
    allowed_ids: Optional[set[int]],
) -> Tuple[Dict[int, np.ndarray], int]:
    class_union: Dict[int, np.ndarray] = {}
    boxes = getattr(result, "boxes", None)
    masks = getattr(result, "masks", None)
    if boxes is None or masks is None:
        return class_union, 0
    if len(boxes) <= 0:
        return class_union, 0

    cls = boxes.cls.detach().cpu().numpy().astype(int)
    conf = boxes.conf.detach().cpu().numpy().astype(np.float32)
    mdat = masks.data.detach().cpu().numpy().astype(np.float32)
    mdat = _resize_masks_to_hw(mdat, H=H, W=W)

    kept = 0
    for i in range(min(len(cls), mdat.shape[0])):
        cid = int(cls[i])
        if float(conf[i]) < float(conf_thr):
            continue
        if allowed_ids is not None and cid not in allowed_ids:
            continue
        mb = mdat[i] > 0.5
        if cid not in class_union:
            class_union[cid] = mb
        else:
            class_union[cid] |= mb
        kept += 1
    return class_union, kept


def _make_overlay(
    image_bgr: np.ndarray,
    class_masks: Dict[int, np.ndarray],
    *,
    class_names: Dict[int, str],
    alpha: float = 0.45,
) -> np.ndarray:
    out = np.asarray(image_bgr, dtype=np.uint8).copy()
    overlay = out.copy()
    alpha = float(np.clip(alpha, 0.0, 1.0))
    colors = [
        (255, 0, 255),
        (0, 255, 0),
        (0, 200, 255),
        (255, 0, 0),
        (255, 180, 0),
        (180, 255, 0),
        (0, 180, 255),
    ]
    for i, cid in enumerate(sorted(class_masks.keys())):
        m = np.asarray(class_masks[cid], dtype=bool)
        col = colors[i % len(colors)]
        overlay[m] = col
    vis = cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0.0)

    y = 22
    for i, cid in enumerate(sorted(class_masks.keys())):
        col = colors[i % len(colors)]
        name = class_names.get(int(cid), f"class_{int(cid)}")
        cv2.putText(
            vis,
            f"{cid}:{name}",
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            col,
            2,
            cv2.LINE_AA,
        )
        y += 22
    return vis


def _show_scene_overlay(
    *,
    window_name: str,
    canvas_bgr: np.ndarray,
    wait_ms: int,
) -> bool:
    """
    Show one scene overlay.

    Returns:
      True  -> continue processing
      False -> stop processing early
    """
    cv2.imshow(window_name, canvas_bgr)
    if int(wait_ms) > 0:
        k = cv2.waitKey(int(wait_ms)) & 0xFF
        if k in (ord("q"), 27):  # q / esc
            return False
        return True

    while True:
        k = cv2.waitKey(0) & 0xFF
        if k in (ord("q"), 27):  # q / esc
            return False
        if k in (ord("n"), ord(" "), 13):  # n / space / enter
            return True


def _downsample_points(points_xyz: np.ndarray, max_points: int) -> np.ndarray:
    pts = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] <= int(max_points):
        return pts
    rng = np.random.default_rng(1234)
    sel = rng.choice(pts.shape[0], size=int(max_points), replace=False)
    return pts[sel]


def _refine_points_for_save(
    points_xyz: np.ndarray,
    *,
    voxel_size: float,
    use_knn_filter: bool,
    knn_k: int,
    knn_std_ratio: float,
    ckdtree_cls: Optional[Any],
) -> np.ndarray:
    pts = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    if pts.shape[0] <= 1:
        return pts
    if float(voxel_size) > 0.0:
        pts = _voxel_downsample_grid(pts, float(voxel_size))
    if bool(use_knn_filter):
        pts = _knn_statistical_filter(
            pts,
            k=int(knn_k),
            std_ratio=float(knn_std_ratio),
            ckdtree_cls=ckdtree_cls,
        )
    return pts


def main() -> None:
    args = _parse_args()
    class_filter = _to_int_set(str(args.class_ids))

    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError("ultralytics is required. Install with `pip install ultralytics`.") from e

    try:
        import create_warning_box_segmentation_images as wbseg
        import test_yolo_pose_from_h5_weights_to_pcd as pose_pcd
    except Exception as e:
        raise RuntimeError("Failed importing project modules. Activate project venv.") from e

    image_paths = _collect_images(str(args.image_path), str(args.image_dir), wbseg=wbseg)
    if int(args.max_images) > 0:
        image_paths = image_paths[: int(args.max_images)]
    if not image_paths:
        raise RuntimeError("No images selected.")

    out_root = Path(str(args.out)).expanduser().resolve()
    out_all = out_root / "all"
    out_cls = out_root / "classes"
    out_vis = out_root / "vis"
    out_all.mkdir(parents=True, exist_ok=True)
    out_cls.mkdir(parents=True, exist_ok=True)
    if int(args.save_vis) == 1:
        out_vis.mkdir(parents=True, exist_ok=True)

    show_vis = bool(int(args.show_vis))
    show_vis_window = str(args.show_vis_window)
    show_vis_ms = int(max(0, int(args.show_vis_ms)))
    vis_enabled = bool(show_vis)
    if vis_enabled:
        print(
            f"[show-vis] enabled=1 window='{show_vis_window}' "
            f"mode={'auto' if show_vis_ms > 0 else 'manual'} wait_ms={show_vis_ms}"
        )
    want_show_pcd = bool(int(args.show_pcd))
    show_pcd_enabled = False
    o3d = None
    if want_show_pcd:
        try:
            import open3d as o3d  # type: ignore

            show_pcd_enabled = True
            print(
                f"[show-pcd] enabled=1 max_points={int(max(1000, int(args.show_pcd_max_points)))} "
                "(close each Open3D window to continue)"
            )
        except Exception as e:
            print(f"[show-pcd] disabled (open3d import failed): {type(e).__name__}: {e}")

    use_knn_filter = bool(int(args.knn_filter))
    expand_radius_m = float(max(0.0, float(args.expand_radius_m)))
    use_expand_radius = bool(expand_radius_m > 0.0)
    ckdtree_cls = None
    if use_knn_filter or use_expand_radius:
        try:
            from scipy.spatial import cKDTree as _cKDTree  # type: ignore

            ckdtree_cls = _cKDTree
            if use_knn_filter:
                print(
                    f"[knn-filter] enabled=1 k={int(max(2, int(args.knn_k)))} "
                    f"std_ratio={float(args.knn_std_ratio):.3f}"
                )
        except Exception as e:
            if use_knn_filter:
                use_knn_filter = False
                print(f"[knn-filter] disabled (scipy unavailable): {type(e).__name__}: {e}")
            if use_expand_radius:
                use_expand_radius = False
                expand_radius_m = 0.0
                print(f"[expand-3d] disabled (scipy unavailable): {type(e).__name__}: {e}")
    if not use_knn_filter:
        print("[knn-filter] enabled=0")
    if use_expand_radius:
        print(f"[expand-3d] enabled=1 radius_m={expand_radius_m:.3f}")
    else:
        print("[expand-3d] enabled=0")

    print(
        f"[mask-cleanup] open_ksize={int(args.mask_open_ksize)} "
        f"close_ksize={int(args.mask_close_ksize)} min_area={int(args.mask_min_area)}"
    )
    if float(args.voxel_size) > 0.0:
        print(f"[voxel] enabled=1 voxel_size={float(args.voxel_size):.6f}")
    else:
        print("[voxel] enabled=0")

    print(f"[input] images={len(image_paths)}")
    print(f"[input] source_h5_root={args.source}")
    print(f"[input] weights={args.weights}")
    if class_filter is not None:
        print(f"[filter] class_ids={sorted(class_filter)}")
    else:
        print("[filter] class_ids=all")

    h5_by_stem = wbseg._build_h5_index(str(args.source))
    if not h5_by_stem:
        raise RuntimeError(f"No H5 files found under: {args.source}")

    model = YOLO(str(Path(str(args.weights)).expanduser().resolve()))
    class_names = _as_class_name_map(getattr(model, "names", {}))

    rows: List[Dict[str, Any]] = []
    saved_all = 0
    saved_cls = 0
    skipped_bad_stem = 0
    skipped_no_h5 = 0
    skipped_no_xyz = 0
    stopped_by_user = False

    for i, ip in enumerate(image_paths, 1):
        try:
            h5_stem, scene_name = wbseg._parse_unique_scene_stem(ip.stem)
        except Exception as e:
            skipped_bad_stem += 1
            print(f"[{i}/{len(image_paths)}] [skip] bad scene stem: {ip.name} ({e})")
            continue
        unique_scene = f"{h5_stem}__{scene_name}"

        h5_matches = h5_by_stem.get(str(h5_stem), [])
        if not h5_matches:
            skipped_no_h5 += 1
            print(f"[{i}/{len(image_paths)}] [skip] no H5 for stem={h5_stem}")
            continue
        h5_path = h5_matches[0]

        xyz_hw3, _mask_air, H, W, reason = wbseg._load_scene_xyz(
            h5_path=h5_path,
            scene_name=str(scene_name),
            pose_mod=pose_pcd,
        )
        if xyz_hw3 is None:
            skipped_no_xyz += 1
            print(f"[{i}/{len(image_paths)}] [skip] xyz missing: {unique_scene} ({reason})")
            continue

        bgr = cv2.imread(str(ip), cv2.IMREAD_COLOR)
        if bgr is None:
            skipped_no_xyz += 1
            print(f"[{i}/{len(image_paths)}] [skip] unreadable image: {ip.name}")
            continue
        if int(bgr.shape[0]) != int(H) or int(bgr.shape[1]) != int(W):
            bgr = cv2.resize(bgr, (int(W), int(H)), interpolation=cv2.INTER_LINEAR)

        pred = model.predict(
            source=bgr,
            conf=float(args.conf),
            iou=float(args.iou),
            imgsz=int(args.imgsz),
            max_det=int(args.max_det),
            device=(str(args.device).strip() if str(args.device).strip() else None),
            verbose=False,
        )
        if not pred:
            print(f"[{i}/{len(image_paths)}] [skip] no prediction object: {unique_scene}")
            continue
        result = pred[0]

        class_union, kept = _build_class_union_masks(
            result=result,
            H=int(H),
            W=int(W),
            conf_thr=float(args.conf),
            allowed_ids=class_filter,
        )
        class_union = _apply_mask_cleanup(
            class_union,
            open_ksize=int(args.mask_open_ksize),
            close_ksize=int(args.mask_close_ksize),
            min_area=int(args.mask_min_area),
        )

        xyz = np.asarray(xyz_hw3, dtype=np.float32)
        finite = np.all(np.isfinite(xyz), axis=2)
        scene_pts_finite = xyz[finite].reshape(-1, 3)
        all_mask = np.zeros((int(H), int(W)), dtype=bool)
        all_scene_keep = np.zeros((scene_pts_finite.shape[0],), dtype=bool)
        row: Dict[str, Any] = {
            "unique_scene": unique_scene,
            "h5_file": str(h5_path.name),
            "scene_name": str(scene_name),
            "image_path": str(ip),
            "pred_instances_kept": int(kept),
            "saved_all_pcd": 0,
            "saved_class_pcd_count": 0,
        }
        class_pts_for_view: Dict[int, np.ndarray] = {}

        for cid in sorted(class_union.keys()):
            m = np.asarray(class_union[cid], dtype=bool) & finite
            all_mask |= m
            pts_seed = xyz[m].reshape(-1, 3)
            if use_expand_radius:
                class_keep, pts_raw = _expand_seeds_to_scene_points(
                    pts_seed,
                    scene_pts_finite,
                    radius_m=float(expand_radius_m),
                    ckdtree_cls=ckdtree_cls,
                )
                if class_keep.shape[0] == all_scene_keep.shape[0]:
                    all_scene_keep |= class_keep
            else:
                pts_raw = pts_seed
            pts = _refine_points_for_save(
                pts_raw,
                voxel_size=float(args.voxel_size),
                use_knn_filter=bool(use_knn_filter),
                knn_k=int(args.knn_k),
                knn_std_ratio=float(args.knn_std_ratio),
                ckdtree_cls=ckdtree_cls,
            )
            class_pts_for_view[int(cid)] = pts
            cname = class_names.get(int(cid), f"class_{int(cid)}")
            cname_clean = _sanitize_name(cname)
            row[f"class_{cid}_name"] = cname
            row[f"class_{cid}_pixels"] = int(np.count_nonzero(m))
            row[f"class_{cid}_points_seed"] = int(pts_seed.shape[0])
            row[f"class_{cid}_points_raw"] = int(pts_raw.shape[0])
            row[f"class_{cid}_points"] = int(pts.shape[0])
            if pts.shape[0] >= int(max(1, args.min_points)):
                pcd_fp = out_cls / cname_clean / f"{unique_scene}.pcd"
                _write_pcd_xyz(pcd_fp, pts)
                row[f"class_{cid}_pcd"] = str(pcd_fp)
                row["saved_class_pcd_count"] = int(row["saved_class_pcd_count"]) + 1
                saved_cls += 1
            else:
                row[f"class_{cid}_pcd"] = ""

        if use_expand_radius:
            pts_all_raw = scene_pts_finite[all_scene_keep].reshape(-1, 3)
        else:
            pts_all_raw = xyz[all_mask & finite].reshape(-1, 3)
        pts_all = _refine_points_for_save(
            pts_all_raw,
            voxel_size=float(args.voxel_size),
            use_knn_filter=bool(use_knn_filter),
            knn_k=int(args.knn_k),
            knn_std_ratio=float(args.knn_std_ratio),
            ckdtree_cls=ckdtree_cls,
        )
        row["all_pixels"] = int(np.count_nonzero(all_mask))
        row["all_points_raw"] = int(pts_all_raw.shape[0])
        row["all_points"] = int(pts_all.shape[0])
        if pts_all.shape[0] >= int(max(1, args.min_points)):
            all_fp = out_all / f"{unique_scene}.pcd"
            _write_pcd_xyz(all_fp, pts_all)
            row["all_pcd"] = str(all_fp)
            row["saved_all_pcd"] = 1
            saved_all += 1
        else:
            row["all_pcd"] = ""

        vis = _make_overlay(
            image_bgr=bgr,
            class_masks=class_union,
            class_names=class_names,
            alpha=0.45,
        )
        if int(args.save_vis) == 1:
            vis_fp = out_vis / f"{unique_scene}.png"
            cv2.imwrite(str(vis_fp), vis)
            row["vis_path"] = str(vis_fp)
        else:
            row["vis_path"] = ""

        rows.append(row)
        print(
            f"[{i}/{len(image_paths)}] [ok] {unique_scene} "
            f"instances={int(kept)} all_points={int(pts_all.shape[0])}"
        )

        if vis_enabled:
            try:
                title = vis.copy()
                cv2.putText(
                    title,
                    f"{i}/{len(image_paths)} {unique_scene}",
                    (8, max(18, int(title.shape[0]) - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cont = _show_scene_overlay(
                    window_name=show_vis_window,
                    canvas_bgr=title,
                    wait_ms=show_vis_ms,
                )
            except Exception as e:
                vis_enabled = False
                print(f"[show-vis] disabled due to GUI error: {type(e).__name__}: {e}")
                cont = True
            if not cont:
                stopped_by_user = True
                print("[show-vis] stopped by user.")
                break

        if show_pcd_enabled and o3d is not None:
            try:
                geoms: List[Any] = []
                pts_main = _downsample_points(
                    pts_all,
                    max_points=int(max(1000, int(args.show_pcd_max_points))),
                )
                if pts_main.shape[0] > 0:
                    pc_main = o3d.geometry.PointCloud()
                    pc_main.points = o3d.utility.Vector3dVector(pts_main.astype(np.float64))
                    pc_main.paint_uniform_color([0.70, 0.70, 0.70])
                    geoms.append(pc_main)

                color_lut = {
                    0: [0.2, 0.4, 1.0],   # aircraft
                    1: [1.0, 0.0, 1.0],   # front gear
                    2: [0.0, 1.0, 0.0],   # engine left
                    3: [1.0, 0.6, 0.0],   # engine right
                }
                for cid in sorted(class_pts_for_view.keys()):
                    pts_c = _downsample_points(
                        class_pts_for_view[int(cid)],
                        max_points=int(max(1000, int(args.show_pcd_max_points))),
                    )
                    if pts_c.shape[0] <= 0:
                        continue
                    pc = o3d.geometry.PointCloud()
                    pc.points = o3d.utility.Vector3dVector(pts_c.astype(np.float64))
                    col = np.asarray(color_lut.get(int(cid), [1.0, 1.0, 0.0]), dtype=np.float64).reshape(1, 3)
                    pc.colors = o3d.utility.Vector3dVector(np.repeat(col, pts_c.shape[0], axis=0))
                    geoms.append(pc)

                if geoms:
                    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0))
                    o3d.visualization.draw_geometries(
                        geoms,
                        window_name=f"Seg->PCD {i}/{len(image_paths)} {unique_scene}",
                        width=1400,
                        height=900,
                    )
            except Exception as e:
                show_pcd_enabled = False
                print(f"[show-pcd] disabled due to viewer error: {type(e).__name__}: {e}")

    summary_fp = out_root / "seg_to_pcd_summary.csv"
    keys: List[str] = []
    seen = set()
    base_order = [
        "unique_scene",
        "h5_file",
        "scene_name",
        "image_path",
        "pred_instances_kept",
        "all_pixels",
        "all_points",
        "all_pcd",
        "saved_all_pcd",
        "saved_class_pcd_count",
        "vis_path",
    ]
    for k in base_order:
        keys.append(k)
        seen.add(k)
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)

    with summary_fp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[out] all_pcd={out_all}")
    print(f"[out] class_pcd={out_cls}")
    if int(args.save_vis) == 1:
        print(f"[out] vis={out_vis}")
    print(f"[out] summary={summary_fp}")
    print(
        "[summary] "
        f"processed={len(rows)} saved_all={saved_all} saved_class={saved_cls} "
        f"skipped_bad_stem={skipped_bad_stem} skipped_no_h5={skipped_no_h5} skipped_no_xyz={skipped_no_xyz}"
    )
    if vis_enabled or bool(show_vis):
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
    if stopped_by_user:
        print("[summary] stopped_by_user=1")


if __name__ == "__main__":
    main()
