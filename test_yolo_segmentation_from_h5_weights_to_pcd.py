#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run a YOLO segmentation model on image input and backproject predicted masks
to point clouds using the matching H5 scene XYZ grid.

Expected image filename stem format:
    <h5_stem>__<scene_name>
Example:
    movement_737_900er__2025-09-11T19-56-15__scene_000.png
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import h5py
import numpy as np

from io_helpers import list_h5_paths, open_h5_any
import test_yolo_pose_from_h5_weights_to_pcd as pose_pcd


DEFAULT_IMAGE_PATH = (
    "/home/femi/yolo_pose_dataset_creation/"
    "warning_box_seg_masks/yolo_seg_dataset/images/val/"
    "movement_737_900er__2025-09-11T19-56-15__scene_000.png"
)
DEFAULT_WEIGHTS_PATH = (
    "/home/femi/Benchmarking_framework/runs/segment/train-4/weights/best.pt"
)
DEFAULT_SOURCE_H5_ROOT = "/home/femi/Benchmarking_framework/Data/warning_b_test_h5"
DEFAULT_OUT_DIR = str(Path.cwd() / "pcd_from_yolo_seg2")
INSTANCE_COLORS_BGR: List[Tuple[int, int, int]] = [
    (80, 220, 255),
    (80, 255, 120),
    (80, 255, 120),
    # (255, 120, 220),
    # (120, 180, 255),
    # (220, 255, 80),
]
INSTANCE_COLORS_RGB_NORM: List[Tuple[float, float, float]] = [
    (1.00, 0.86, 0.31),
    (0.31, 1.00, 0.47),
    (0.31, 1.00, 0.47),
    # (1.00, 0.47, 0.86),
    # (1.00, 0.65, 0.31),
    # (0.86, 1.00, 0.31),
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Run a YOLO segmentation model on image(s), then transfer the "
            "predicted mask to point cloud using the matching H5 scene."
        )
    )
    ap.add_argument(
        "--image-path",
        type=str,
        default="",
        help="Single image path or directory path",
    )
    ap.add_argument(
        "--image-dir",
        type=str,
        default="/home/femi/yolo_pose_dataset_creation/aircraft_pose_with_normalising_applied_multifield_only_3_2/images/test",
        help="Directory containing test images",
    )
    ap.add_argument(
        "--weights",
        type=str,
        default=DEFAULT_WEIGHTS_PATH,
        help="YOLO segmentation .pt weights",
    )
    ap.add_argument(
        "--source",
        type=str,
        default=DEFAULT_SOURCE_H5_ROOT,
        help="Root folder containing source H5 files",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=DEFAULT_OUT_DIR,
        help="Output directory",
    )
    ap.add_argument("--imgsz", type=int, default=1024, help="YOLO inference image size")
    ap.add_argument("--conf", type=float, default=0.05, help="YOLO confidence threshold")
    ap.add_argument("--device", type=str, default="0", help="YOLO device (e.g. 0 or cpu)")
    # ap.add_argument(
    #     "--target-class",
    #     type=int,
    #     default="",
    #     help="Segmentation class id to export",
    # )
    ap.add_argument(
        "--target-classes",
        type=str,
        default="0,1,2",
        help="Comma-separated class ids to merge into one point cloud, e.g. 1,2,3",
    )
    ap.add_argument(
        "--instance-mode",
        type=str,
        default="merge",
        choices=["best", "merge"],
        help=(
            "best: save the highest-confidence instance of target class. "
            "merge: union all predicted instances of target class into one mask."
        ),
    )
    ap.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Maximum images to process (0 = all)",
    )
    ap.add_argument(
        "--save-debug-image",
        type=str,
        default="on",
        choices=["on", "off"],
        help="Save debug image overlays under <out>/debug_imgs",
    )
    ap.add_argument(
        "--save-mask-png",
        type=str,
        default="on",
        choices=["on", "off"],
        help="Save predicted binary masks under <out>/mask_png",
    )
    ap.add_argument(
        "--visualize",
        type=str,
        default="on",
        choices=["on", "off"],
        help="Show Open3D visualization for each exported segmentation point cloud.",
    )
    ap.add_argument(
        "--visualize-image",
        type=str,
        default="off",
        choices=["on", "off"],
        help="Show a 2D image window with the predicted segmentation overlay.",
    )
    return ap.parse_args()


def _to_bool_switch(raw: str, fallback: bool) -> bool:
    s = str(raw or "").strip().lower()
    if s in {"on", "true", "1", "yes", "y"}:
        return True
    if s in {"off", "false", "0", "no", "n"}:
        return False
    return bool(fallback)


def _load_scene_points_and_meta(grp: h5py.Group) -> Tuple[np.ndarray, List[str]]:
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


def _parse_target_classes(raw: str, fallback_class: Optional[int] = None) -> List[int]:
    s = str(raw or "").strip()
    if not s:
        if fallback_class is None:
            return []
        return [int(fallback_class)]
    out: List[int] = []
    seen: set[int] = set()
    for part in s.split(","):
        t = str(part).strip()
        if not t:
            continue
        cid = int(t)
        if cid in seen:
            continue
        seen.add(cid)
        out.append(cid)
    if not out:
        if fallback_class is None:
            return []
        return [int(fallback_class)]
    return out


def _select_segmentation_mask(
    result: Any,
    *,
    target_classes: List[int],
    instance_mode: str,
    out_h: int,
    out_w: int,
) -> Optional[Tuple[np.ndarray, float, int, List[Dict[str, Any]]]]:
    boxes = getattr(result, "boxes", None)
    masks = getattr(result, "masks", None)
    if boxes is None or masks is None or len(boxes) == 0:
        return None
    if getattr(masks, "data", None) is None:
        return None

    cls = boxes.cls.detach().cpu().numpy().astype(int)
    conf = boxes.conf.detach().cpu().numpy().astype(float)
    mask_data = masks.data.detach().cpu().numpy()

    if mask_data.ndim != 3 or mask_data.shape[0] != len(cls):
        return None

    keep_set = {int(v) for v in target_classes}
    keep_idx = [i for i in range(len(cls)) if int(cls[i]) in keep_set]
    if not keep_idx:
        return None

    def _resize_mask(mask_like: np.ndarray) -> np.ndarray:
        mask_u8 = np.asarray(mask_like, dtype=np.uint8)
        if mask_u8.shape[0] != int(out_h) or mask_u8.shape[1] != int(out_w):
            mask_u8 = cv2.resize(
                mask_u8,
                (int(out_w), int(out_h)),
                interpolation=cv2.INTER_NEAREST,
            )
        return np.asarray(mask_u8 > 0, dtype=bool)

    if str(instance_mode) == "merge":
        chosen_idx = list(keep_idx)
    else:
        best_i = max(keep_idx, key=lambda i: float(conf[i]))
        chosen_idx = [int(best_i)]

    instance_infos: List[Dict[str, Any]] = []
    merged_mask = np.zeros((int(out_h), int(out_w)), dtype=bool)
    for j, idx in enumerate(chosen_idx):
        mask_hw = _resize_mask(mask_data[idx] > 0.5)
        merged_mask |= mask_hw
        instance_infos.append(
            {
                "instance_idx": int(j),
                "class_id": int(cls[idx]),
                "score": float(conf[idx]),
                "mask_hw": mask_hw,
            }
        )

    score = max(float(conf[i]) for i in chosen_idx)
    count = len(chosen_idx)
    return merged_mask, float(score), int(count), instance_infos


def _draw_mask_overlay(
    image_bgr: np.ndarray,
    mask_hw: np.ndarray,
    *,
    class_label: str,
    score: float,
    instance_count: int,
    instance_infos: Optional[List[Dict[str, Any]]] = None,
) -> np.ndarray:
    out = image_bgr.copy()
    infos = list(instance_infos or [])
    if infos:
        for info in infos:
            m = np.asarray(info.get("mask_hw", None), dtype=bool)
            if m.shape[:2] != out.shape[:2]:
                continue
            col = INSTANCE_COLORS_BGR[int(info.get("instance_idx", 0)) % len(INSTANCE_COLORS_BGR)]
            tint = np.zeros_like(out)
            tint[:, :] = np.asarray(col, dtype=np.uint8)
            out[m] = cv2.addWeighted(out[m], 0.35, tint[m], 0.65, 0.0)
    else:
        m = np.asarray(mask_hw, dtype=bool)
        tint = np.zeros_like(out)
        tint[:, :, 1] = 255
        tint[:, :, 2] = 120
        out[m] = cv2.addWeighted(out[m], 0.35, tint[m], 0.65, 0.0)
    return out


def _show_open3d_segmentation_points(
    pts_all: np.ndarray,
    pts_mask: np.ndarray,
    *,
    window_name: str,
    instance_points: Optional[List[Dict[str, Any]]] = None,
) -> bool:
    try:
        import open3d as o3d
    except Exception as e:
        print(f"  [WARN] Open3D not available, skipping 3D view ({e})")
        return False

    geoms = []

    if pts_all.size > 0:
        pcd_all = o3d.geometry.PointCloud()
        pcd_all.points = o3d.utility.Vector3dVector(pts_all.astype(np.float64))
        pcd_all.paint_uniform_color([0.68, 0.68, 0.68])
        geoms.append(pcd_all)

    inst_pts = list(instance_points or [])
    if inst_pts:
        for info in inst_pts:
            pts_i = np.asarray(info.get("points_xyz", np.empty((0, 3), dtype=np.float32)), dtype=np.float32).reshape(-1, 3)
            if pts_i.shape[0] <= 0:
                continue
            pcd_i = o3d.geometry.PointCloud()
            pcd_i.points = o3d.utility.Vector3dVector(pts_i.astype(np.float64))
            rgb = INSTANCE_COLORS_RGB_NORM[int(info.get("instance_idx", 0)) % len(INSTANCE_COLORS_RGB_NORM)]
            pcd_i.paint_uniform_color(list(rgb))
            geoms.append(pcd_i)
    elif pts_mask.size > 0:
        pcd_mask = o3d.geometry.PointCloud()
        pcd_mask.points = o3d.utility.Vector3dVector(pts_mask.astype(np.float64))
        pcd_mask.paint_uniform_color([1.0, 0.15, 0.15])
        geoms.append(pcd_mask)

    if not geoms:
        return False

    o3d.visualization.draw_geometries(
        geoms,
        window_name=window_name,
        width=1280,
        height=720,
    )
    return True


def _show_image_overlay(
    image_bgr: np.ndarray,
    *,
    window_name: str,
) -> bool:
    try:
        cv2.imshow(window_name, image_bgr)
        cv2.waitKey(1)
        return True
    except Exception as e:
        print(f"  [WARN] OpenCV image window not available, skipping 2D view ({e})")
        return False


def _write_summary_csv(rows: List[Dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "unique_scene",
        "h5_file",
        "scene_name",
        "image_path",
        "status",
        "reason",
        "target_class",
        "target_classes",
        "instance_mode",
        "instance_count",
        "score",
        "mask_pixels",
        "pcd_points",
        "pcd_path",
        "debug_image_path",
        "mask_png_path",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main() -> None:
    args = parse_args()

    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError("ultralytics is required. Install with `pip install ultralytics`.") from e

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

    save_debug_image = _to_bool_switch(str(args.save_debug_image), True)
    save_mask_png = _to_bool_switch(str(args.save_mask_png), True)
    visualize = _to_bool_switch(str(args.visualize), False)
    visualize_image = _to_bool_switch(str(args.visualize_image), False)
    fallback_target_class = getattr(args, "target_class", None)
    target_classes = _parse_target_classes(
        str(getattr(args, "target_classes", "")),
        (int(fallback_target_class) if fallback_target_class is not None else None),
    )
    if not target_classes:
        raise RuntimeError(
            "No target classes configured. Set --target-classes, for example 1,2,3."
        )
    class_label = ",".join(str(v) for v in target_classes)
    instance_mode = str(args.instance_mode)
    if len(target_classes) > 1 and instance_mode == "best":
        print(
            f"[warn] target_classes={class_label} with instance_mode=best keeps only one mask; "
            "switching to merge"
        )
        instance_mode = "merge"

    out_root = Path(out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    dbg_root = out_root / "debug_imgs"
    mask_root = out_root / "mask_png"
    if save_debug_image:
        dbg_root.mkdir(parents=True, exist_ok=True)
    if save_mask_png:
        mask_root.mkdir(parents=True, exist_ok=True)

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
        f"[pipeline] target_classes={class_label} "
        f"instance_mode={instance_mode}"
    )

    print("[list] Searching for .h5 files...")
    h5_paths = list_h5_paths(source_root)
    if not h5_paths:
        raise RuntimeError(f"No .h5 files found under: {source_root}")

    by_h5_stem: Dict[str, List[str]] = defaultdict(list)
    for hp in h5_paths:
        by_h5_stem[Path(hp).stem].append(hp)

    weights_path = pose_pcd._resolve_weights(weights_raw)
    print(f"[model] using weights: {weights_path}")
    model = YOLO(str(weights_path))

    summary_rows: List[Dict[str, Any]] = []
    total_saved_pcd = 0
    total_saved_dbg = 0
    total_saved_masks = 0

    for idx, image_path in enumerate(image_paths, 1):
        row: Dict[str, Any] = {
            "unique_scene": "",
            "h5_file": "",
            "scene_name": "",
            "image_path": str(image_path),
            "status": "FAIL",
            "reason": "",
            "target_class": (
                int(fallback_target_class) if fallback_target_class is not None else ""
            ),
            "target_classes": class_label,
            "instance_mode": instance_mode,
            "instance_count": 0,
            "score": "",
            "mask_pixels": 0,
            "pcd_points": 0,
            "pcd_path": "",
            "debug_image_path": "",
            "mask_png_path": "",
        }

        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            row["reason"] = "unreadable_image"
            summary_rows.append(row)
            print(f"[{idx}/{len(image_paths)}] [skip] unreadable image: {image_path}")
            continue

        try:
            h5_stem, scene_name = pose_pcd._parse_unique_scene_stem(image_path.stem)
        except Exception as e:
            row["reason"] = f"bad_image_stem:{type(e).__name__}"
            summary_rows.append(row)
            print(f"[{idx}/{len(image_paths)}] [skip] bad scene stem: {image_path.name} ({e})")
            continue

        unique_scene = f"{h5_stem}__{scene_name}"
        row["unique_scene"] = unique_scene
        row["scene_name"] = str(scene_name)

        matches = by_h5_stem.get(h5_stem, [])
        if not matches:
            row["reason"] = "no_matching_h5"
            summary_rows.append(row)
            print(f"[{idx}/{len(image_paths)}] [skip] no H5 for stem={h5_stem}")
            continue
        if len(matches) > 1:
            print(f"[warn] multiple H5 matches for '{h5_stem}', using first: {matches[0]}")
        h5_path = matches[0]
        row["h5_file"] = Path(h5_path).name

        try:
            with open_h5_any(h5_path) as f:
                H = int(f.attrs["height"])
                W = int(f.attrs["width"])
                if scene_name not in f:
                    row["reason"] = "scene_not_in_h5"
                    summary_rows.append(row)
                    print(f"[{idx}/{len(image_paths)}] [skip] scene not in H5: {unique_scene}")
                    continue

                grp = f[scene_name]
                if not isinstance(grp, h5py.Group) or "points" not in grp:
                    row["reason"] = "scene_has_no_points"
                    summary_rows.append(row)
                    print(f"[{idx}/{len(image_paths)}] [skip] scene has no points: {unique_scene}")
                    continue

                flat, cols = _load_scene_points_and_meta(grp)
                rgb_h5, xyz_hw3 = pose_pcd._build_rgb_and_xyz(flat, cols, H, W)
                if rgb_h5 is None or xyz_hw3 is None:
                    row["reason"] = "missing_required_xyz_cols"
                    summary_rows.append(row)
                    print(f"[{idx}/{len(image_paths)}] [skip] cannot build xyz/rgb: {unique_scene}")
                    continue

                mask_aircraft = pose_pcd._extract_is_aircraft_mask(flat, cols, H, W)
                if mask_aircraft is not None:
                    shift_cols = pose_pcd._compute_export_like_roll(mask_aircraft)
                    if shift_cols != 0:
                        xyz_hw3 = np.roll(xyz_hw3, shift=shift_cols, axis=1)
                        rgb_h5 = np.roll(rgb_h5, shift=shift_cols, axis=1)
                        mask_aircraft = np.roll(mask_aircraft, shift=shift_cols, axis=1)

                rgb_for_model = bgr
                if rgb_for_model.shape[0] != H or rgb_for_model.shape[1] != W:
                    print(
                        f"[{idx}/{len(image_paths)}] [image] resize "
                        f"{rgb_for_model.shape[1]}x{rgb_for_model.shape[0]} -> {W}x{H} ({unique_scene})"
                    )
                    rgb_for_model = cv2.resize(
                        rgb_for_model,
                        (W, H),
                        interpolation=cv2.INTER_LINEAR,
                    )

                results = model.predict(
                    rgb_for_model,
                    imgsz=int(args.imgsz),
                    conf=float(args.conf),
                    device=str(args.device),
                    verbose=False,
                )
                if not results:
                    row["reason"] = "no_yolo_result"
                    summary_rows.append(row)
                    print(f"[{idx}/{len(image_paths)}] [skip] no result: {unique_scene}")
                    continue

                selected = _select_segmentation_mask(
                    results[0],
                    target_classes=target_classes,
                    instance_mode=instance_mode,
                    out_h=int(H),
                    out_w=int(W),
                )
                if selected is None:
                    row["reason"] = "no_target_class_mask"
                    summary_rows.append(row)
                    print(f"[{idx}/{len(image_paths)}] [skip] no mask for class set {class_label}: {unique_scene}")
                    continue

                mask_hw, score, instance_count, instance_infos = selected
                pts = np.asarray(xyz_hw3[mask_hw], dtype=np.float32).reshape(-1, 3)
                finite = np.all(np.isfinite(pts), axis=1)
                pts = pts[finite]
                pts_all = np.asarray(xyz_hw3, dtype=np.float32).reshape(-1, 3)
                pts_all = pts_all[np.all(np.isfinite(pts_all), axis=1)]
                instance_points: List[Dict[str, Any]] = []
                for info in instance_infos:
                    m_i = np.asarray(info.get("mask_hw", None), dtype=bool)
                    if m_i.shape[:2] != xyz_hw3.shape[:2]:
                        continue
                    pts_i = np.asarray(xyz_hw3[m_i], dtype=np.float32).reshape(-1, 3)
                    pts_i = pts_i[np.all(np.isfinite(pts_i), axis=1)]
                    instance_points.append(
                        {
                            "instance_idx": int(info.get("instance_idx", 0)),
                            "class_id": int(info.get("class_id", -1)),
                            "score": float(info.get("score", 0.0)),
                            "points_xyz": pts_i,
                            "mask_hw": m_i,
                        }
                    )
                if pts.shape[0] <= 0:
                    row["reason"] = "empty_pointcloud_from_mask"
                    row["score"] = f"{float(score):.6f}"
                    row["instance_count"] = int(instance_count)
                    row["mask_pixels"] = int(np.count_nonzero(mask_hw))
                    summary_rows.append(row)
                    print(f"[{idx}/{len(image_paths)}] [skip] empty pointcloud from mask: {unique_scene}")
                    continue

                out_pcd = out_root / f"{unique_scene}.pcd"
                pose_pcd.write_pcd_xyz(out_pcd, pts)

                row["status"] = "PASS"
                row["reason"] = "ok"
                row["score"] = f"{float(score):.6f}"
                row["instance_count"] = int(instance_count)
                row["mask_pixels"] = int(np.count_nonzero(mask_hw))
                row["pcd_points"] = int(pts.shape[0])
                row["pcd_path"] = str(out_pcd)
                total_saved_pcd += 1

                if save_mask_png:
                    mask_fp = mask_root / f"{unique_scene}.png"
                    cv2.imwrite(str(mask_fp), np.asarray(mask_hw, dtype=np.uint8) * 255)
                    row["mask_png_path"] = str(mask_fp)
                    total_saved_masks += 1

                if save_debug_image:
                    dbg = _draw_mask_overlay(
                        image_bgr=rgb_for_model,
                        mask_hw=mask_hw,
                        class_label=class_label,
                        score=float(score),
                        instance_count=int(instance_count),
                        instance_infos=instance_infos,
                    )
                    dbg_fp = dbg_root / f"{unique_scene}.png"
                    cv2.imwrite(str(dbg_fp), dbg)
                    row["debug_image_path"] = str(dbg_fp)
                    total_saved_dbg += 1
                else:
                    dbg = _draw_mask_overlay(
                        image_bgr=rgb_for_model,
                        mask_hw=mask_hw,
                        class_label=class_label,
                        score=float(score),
                        instance_count=int(instance_count),
                        instance_infos=instance_infos,
                    )

                img_window_name = f"YOLO-Seg Overlay: {unique_scene}"
                if visualize_image:
                    _show_image_overlay(
                        dbg,
                        window_name=img_window_name,
                    )

                if visualize:
                    _show_open3d_segmentation_points(
                        pts_all=pts_all,
                        pts_mask=pts,
                        window_name=f"YOLO-Seg -> PCD: {unique_scene}",
                        instance_points=instance_points,
                    )
                    if visualize_image:
                        try:
                            cv2.destroyWindow(img_window_name)
                            cv2.waitKey(1)
                        except Exception:
                            pass

                summary_rows.append(row)
                print(
                    f"[{idx}/{len(image_paths)}] [ok] {unique_scene}: "
                    f"mask_pixels={int(np.count_nonzero(mask_hw))} "
                    f"pcd_points={int(pts.shape[0])} "
                    f"score={float(score):.3f} "
                    f"instances={int(instance_count)}"
                )

        except Exception as e:
            row["reason"] = f"{type(e).__name__}:{e}"
            summary_rows.append(row)
            print(f"[{idx}/{len(image_paths)}] [error] {unique_scene}: {type(e).__name__}: {e}")

    summary_csv = out_root / "segmentation_to_pcd_summary.csv"
    _write_summary_csv(summary_rows, summary_csv)

    print(f"[summary] summary_csv={summary_csv}")
    print(f"[summary] pcd_saved={int(total_saved_pcd)}")
    print(f"[summary] debug_images_saved={int(total_saved_dbg)}")
    print(f"[summary] mask_png_saved={int(total_saved_masks)}")
    print(f"[summary] total_images={len(image_paths)}")


if __name__ == "__main__":
    main()
