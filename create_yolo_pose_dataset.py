#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create a unified YOLO-pose dataset from many aircraft HDF5 LiDAR range-image files.

This script:
- Reads all HDF5 scenes (local or gs://)
- Projects 3D aircraft keypoints
- Creates label files + images for YOLO pose
- Adds synthetic midpoint keypoint ("front_wheels_mid")
- Applies visibility + cluster adjustment rules
- Produces train/val/test splits
- Writes YOLO dataset YAML and visualizations

Uses defaults from config_dataset.py, but you can override via CLI:

    python create_yolo_pose_dataset.py \\
        --source gs://.../mydata \\
        --out ./my_dataset \\
        --train-ratio 0.7 --val-ratio 0.2 --test-ratio 0.1 \\
        --max-h5 5
"""

import json
import argparse
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from collections import defaultdict

import numpy as np
import h5py
import imageio.v2 as imageio
import cv2

from config_dataset import (
    SOURCE,
    OUT_DIR,
    SPLIT,
    RANDOM_SEED,
    MAX_H5_FILES,
    DRAW_ON_OVERLAY,
    APPLY_MEDIAN_FILTER,
    MEDIAN_KSIZE,
    USE_TF_MATRIX,
    APPLY_Z_FLIP,
    SYN_KP_NAME,
    REMOVE_KP_SET,
    FRONT_RIGHT_ALIASES,
    FRONT_LEFT_ALIASES,
    MAKE_VIZ,
    KPT_BBOX_MARGIN_PX,
    ROLL_WIDE_BBOX,
    ROLL_WIDE_BBOX_FRAC,
    ROLL_WIDE_BBOX_COLS,  # still kept for diagnostics / compatibility
    RAY_VISIBILITY_CHECK,
    RAY_TOL,
    RAY_PATCH_RADIUS,
    MID_BASE_RADIUS,
    MID_EXPAND_RADIUS,
    MID_Z_BAND,
    MID_MIN_POINTS,
)

from io_helpers import list_h5_paths, open_h5_any
from projection_helpers import (
    build_rgb_from_cols,
    bbox_from_mask,
    xyxy_to_xywhn,
    get_min_depth,
    adjust_midpoint_to_cluster,
    angles_from_xyz,
    row_from_elevation,
    col_from_azimuth_global,
    apply_transform,
    rc_to_xy_norm,
    find_alias,
    _fill_nans,
)


# =========================
# Helper: smart azimuth seam placement
# =========================
def find_best_azimuth_roll(mask2d: np.ndarray) -> int:
    """
    Given a boolean is_aircraft mask (H, W), find a horizontal roll (shift)
    that places the panorama seam (column 0) in the largest empty-azimuth gap.

    Returns:
        shift (int): number of columns to np.roll(..., shift=shift, axis=1)

    If we cannot find any empty columns, or rolling does not help, returns 0.
    """
    H, W = mask2d.shape
    col_has_aircraft = mask2d.any(axis=0)  # (W,)
    empty = ~col_has_aircraft

    if not np.any(empty):
        # Every column has aircraft → nothing to do
        return 0

    best_len = 0
    best_start = 0

    # Simple O(W^2) search, W~1024 so fine.
    for start in range(W):
        if not empty[start]:
            continue
        length = 0
        while length < W and empty[(start + length) % W]:
            length += 1
        if length > best_len:
            best_len = length
            best_start = start

    if best_len <= 0:
        return 0

    # Put seam in the middle of the largest empty run
    seam_col = (best_start + best_len // 2) % W

    # We want seam_col -> 0 (left edge), so roll left by seam_col
    shift = -int(seam_col)
    if shift % W == 0:
        return 0
    return shift


def create_dataset(
    source: str,
    out_dir: str,
    split: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    max_h5_files: Optional[int] = None,
) -> None:
    """
    Create YOLO-pose dataset from aircraft HDF5 LiDAR range-image data.

    Args:
        source (str): root directory or gs:// bucket
        out_dir (str): output dataset folder
        split (tuple): (train, val, test) ratios
        max_h5_files (int or None): limit number of HDF5 files
    """

    if abs(sum(split) - 1.0) >= 1e-6:
        raise ValueError(f"split must sum to 1.0 (got {sum(split):.4f})")

    random.seed(RANDOM_SEED)
    out_root = Path(out_dir)
    for p in [
        "images/train",
        "images/val",
        "images/test",
        "labels/train",
        "labels/val",
        "labels/test",
    ]:
        (out_root / p).mkdir(parents=True, exist_ok=True)

    # vis root
    vis_root = out_root / "vis"
    for p in ["train", "val", "test"]:
        (vis_root / p).mkdir(parents=True, exist_ok=True)

    print("[list] Searching for .h5 files…")
    h5_paths = list_h5_paths(source)
    if not h5_paths:
        raise RuntimeError(f"No .h5 files found under: {source}")

    print(f"[list] Found {len(h5_paths)} HDF5 files total.")

    # Limit number for testing
    if max_h5_files is not None and max_h5_files > 0 and len(h5_paths) > max_h5_files:
        h5_paths = h5_paths[:max_h5_files]
        print(
            f"[list] Using only first {len(h5_paths)} HDF5 files (max_h5_files={max_h5_files})"
        )

    # ==================================================
    # Phase 1 — scan and build unified KEYPOINT ORDER
    # ==================================================

    all_scenes: List[Tuple[str, str]] = []
    KP_ORDER: List[str] = []
    seen_kps = set()
    per_scene_names: Dict[Tuple[str, str], List[str]] = {}

    print("\n--- Phase 1: Scanning files for scenes/keypoints ---")
    for i, h5p in enumerate(h5_paths, 1):
        print(f"[{i}/{len(h5_paths)}] {Path(h5p).name}")
        try:
            with open_h5_any(h5p) as f:
                for s, g in f.items():
                    if isinstance(g, h5py.Group) and "points" in g and "keypoints" in g:
                        all_scenes.append((h5p, s))
                        raw = g["keypoints"]["names"][()]
                        names = [
                            n.decode("utf-8") if isinstance(n, (bytes, bytearray)) else str(n)
                            for n in raw
                        ]
                        per_scene_names[(h5p, s)] = names
                        for n in names:
                            if n in REMOVE_KP_SET:
                                continue
                            if n not in seen_kps:
                                KP_ORDER.append(n)
                                seen_kps.add(n)
        except Exception as e:
            print(f"  [WARN] Skipping file due to error: {e}")

    if SYN_KP_NAME not in KP_ORDER:
        KP_ORDER.append(SYN_KP_NAME)

    if not all_scenes:
        raise RuntimeError("No valid scenes found.")

    print(f"\n[index] Scenes: {len(all_scenes)}")
    print(f"[kps] Unified KP_ORDER ({len(KP_ORDER)}): {', '.join(KP_ORDER)}")

    # Shuffle & split scenes
    random.shuffle(all_scenes)
    n = len(all_scenes)
    n_train = int(n * split[0])
    n_val = int(n * split[1])
    sets = {
        "train": set(all_scenes[:n_train]),
        "val": set(all_scenes[n_train : n_train + n_val]),
        "test": set(all_scenes[n_train + n_val :]),
    }

    # Group scenes by file
    scenes_by_file: Dict[str, List[str]] = defaultdict(list)
    for h5p, s in all_scenes:
        scenes_by_file[h5p].append(s)

    # ==================================================
    # Phase 2 — export dataset
    # ==================================================

    print("\n--- Phase 2: Exporting images/labels (grouped by file) ---")
    total_files = len(scenes_by_file)

    total_valid_scenes = 0
    files_with_valid_scenes = 0

    for fi, (h5p, scene_list) in enumerate(scenes_by_file.items(), 1):
        print(f"[{fi}/{total_files}] {Path(h5p).name}  scenes={len(scene_list)}")

        file_valid_scenes = 0

        try:
            with open_h5_any(h5p) as f:
                H = int(f.attrs["height"])
                W = int(f.attrs["width"])

                for scene_name in scene_list:
                    file_stem = Path(h5p).stem
                    unique_scene = f"{file_stem}__{scene_name}"
                    split_name = (
                        "train"
                        if (h5p, scene_name) in sets["train"]
                        else ("val" if (h5p, scene_name) in sets["val"] else "test")
                    )
                    print(f"  - {unique_scene} → {split_name}")

                    grp = f[scene_name]
                    ds = grp["points"]
                    flat = ds[()]  # (H*W, C)

                    # --- columns ---
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
                    if not {"x", "y", "z"}.issubset(set(cols)):
                        print("    [SKIP] Missing x/y/z")
                        continue
                    ix, iy, iz = cols.index("x"), cols.index("y"), cols.index("z")

                    if "is_aircraft" not in cols:
                        print("    [SKIP] No is_aircraft")
                        continue
                    mask2d = (
                        flat[:, cols.index("is_aircraft")]
                        .astype(np.uint8)
                        .reshape(H, W)
                        .astype(bool)
                    )

                    # --- image ---
                    rgb = build_rgb_from_cols(flat, cols, H, W)
                    if rgb is None:
                        gray = (mask2d.astype(np.uint8) * 255)
                        rgb = np.dstack([gray, gray, gray])
                    img = rgb

                    # optional overlay
                    if DRAW_ON_OVERLAY:
                        overlay = img.copy()
                        red = np.zeros_like(overlay)
                        red[..., 0] = 255
                        alpha = 0.5
                        overlay[mask2d] = (
                            alpha * red[mask2d] + (1 - alpha) * overlay[mask2d]
                        ).astype(np.uint8)
                        img = overlay

                    # optional median filter
                    if APPLY_MEDIAN_FILTER:
                        if img.dtype != np.uint8:
                            img = img.astype(np.uint8)
                        img = cv2.medianBlur(img, MEDIAN_KSIZE)

                    # xyz grid (LiDAR frame)
                    xyz = np.stack(
                        [flat[:, ix], flat[:, iy], flat[:, iz]], axis=1
                    ).astype(np.float64)
                    xyz_hw3 = xyz.reshape(H, W, 3)

                    # --- bbox from is_aircraft mask (BEFORE possible roll) ---
                    bb = bbox_from_mask(mask2d)
                    if bb is None:
                        print("    [SKIP] Empty aircraft mask")
                        continue
                    x1, y1, x2, y2 = bb
                    bbox_w = (x2 - x1 + 1)
                    bbox_frac = bbox_w / float(W)

                    # --- ROLL LOGIC (smart seam placement) ---
                    if ROLL_WIDE_BBOX and bbox_frac > ROLL_WIDE_BBOX_FRAC and W > 1:
                        # Prefer data-driven seam placement
                        shift = find_best_azimuth_roll(mask2d)
                        if shift == 0:
                            # Fallback to config roll if seam logic fails
                            shift = ROLL_WIDE_BBOX_COLS % W

                        if shift != 0:
                            print(
                                f"    [ROLL] Wide bbox (frac={bbox_frac:.3f}) "
                                f"→ rolling by {shift} cols"
                            )
                            img = np.roll(img, shift=shift, axis=1)
                            mask2d = np.roll(mask2d, shift=shift, axis=1)
                            xyz_hw3 = np.roll(xyz_hw3, shift=shift, axis=1)

                            # recompute bbox after roll
                            bb2 = bbox_from_mask(mask2d)
                            if bb2 is None:
                                print("    [SKIP] Empty aircraft mask after roll")
                                continue
                            x1, y1, x2, y2 = bb2
                            bbox_w = (x2 - x1 + 1)
                            bbox_frac = bbox_w / float(W)

                        # still too wide → probably something really off, skip
                        if bbox_frac > 0.6:
                            print(
                                f"    [SKIP] BBox too wide ({bbox_frac:.3f} > 0.6) in {unique_scene}"
                            )
                            continue

                    # normalized bbox
                    cx, cy, bw, bh = xyxy_to_xywhn(x1, y1, x2, y2, W, H)

                    # aircraft points and ground z
                    aircraft_pts = xyz_hw3[mask2d]  # (Na, 3)
                    if aircraft_pts.size > 0:
                        z_min_air = float(np.min(aircraft_pts[:, 2]))
                    else:
                        z_min_air = 0.0

                    # range image for ray-tracing visibility
                    range_img = np.linalg.norm(xyz_hw3, axis=2)  # (H, W)
                    valid_range = np.isfinite(range_img) & mask2d

                    # Vectorized elevation/azimuth calibration
                    valid = np.all(np.isfinite(xyz_hw3), axis=2)
                    el_all = np.full((H, W), np.nan, dtype=np.float64)
                    az_all = np.full((H, W), np.nan, dtype=np.float64)
                    xv = xyz_hw3[..., 0][valid]
                    yv = xyz_hw3[..., 1][valid]
                    zv = xyz_hw3[..., 2][valid]
                    el_all[valid] = np.arctan2(zv, np.sqrt(xv * xv + yv * yv))
                    az_all[valid] = np.arctan2(yv, xv)
                    el_per_row_raw = np.nanmedian(el_all, axis=1)  # (H,)
                    sin_c = np.nanmean(np.sin(az_all), axis=0)
                    cos_c = np.nanmean(np.cos(az_all), axis=0)
                    az_per_col_raw = np.arctan2(sin_c, cos_c)  # (W,)
                    el_per_row_calib = _fill_nans(el_per_row_raw)
                    az_per_col_calib = _fill_nans(az_per_col_raw)

                    # keypoints from file
                    kp_grp = grp["keypoints"]
                    kps_model = np.asarray(kp_grp["xyz"][()], dtype=np.float64)
                    raw_names = kp_grp.get("names", None)
                    scene_names_list = [
                        n.decode("utf-8") if isinstance(n, (bytes, bytearray)) else str(n)
                        for n in (raw_names[()] if raw_names is not None else [])
                    ]
                    if (
                        kps_model.ndim != 2
                        or kps_model.shape[1] != 3
                        or kps_model.shape[0] == 0
                    ):
                        print("    [SKIP] Keypoints malformed")
                        continue

                    ok_rows = np.all(np.isfinite(kps_model), axis=1)
                    kps_model = kps_model[ok_rows]
                    if scene_names_list:
                        scene_names_list = [
                            scene_names_list[i] for i, t in enumerate(ok_rows) if t
                        ]
                    else:
                        scene_names_list = [f"k{i}" for i in range(kps_model.shape[0])]

                    name_to_idx_full = {n: i for i, n in enumerate(scene_names_list)}

                    # 3D midpoint from two wheel-link KPs
                    mid_raw_3d = None
                    nm_fr = find_alias(scene_names_list, FRONT_RIGHT_ALIASES)
                    nm_fl = find_alias(scene_names_list, FRONT_LEFT_ALIASES)
                    if (nm_fr in name_to_idx_full) and (nm_fl in name_to_idx_full):
                        p_fr = kps_model[name_to_idx_full[nm_fr]]
                        p_fl = kps_model[name_to_idx_full[nm_fl]]
                        mid_raw_3d = 0.5 * (p_fr + p_fl)

                    # remove unwanted KPs
                    keep_mask = [n not in REMOVE_KP_SET for n in scene_names_list]
                    names_kept = [n for n, keep in zip(scene_names_list, keep_mask) if keep]
                    kps_kept = kps_model[keep_mask]
                    name_to_idx_kept = {n: i for i, n in enumerate(names_kept)}

                    # apply transform only to base_link
                    T = None
                    if USE_TF_MATRIX and "metadata" in grp and "tf_matrix" in grp["metadata"]:
                        T_ = np.asarray(grp["metadata"]["tf_matrix"][()], dtype=np.float64)
                        if T_.shape == (4, 4) and np.all(np.isfinite(T_)):
                            T = T_ if not APPLY_Z_FLIP else (T_ @ np.diag([1.0, 1.0, -1.0, 1.0]))
                    kps_scene = kps_kept.copy()
                    if ("base_link" in name_to_idx_kept) and (T is not None):
                        i_base = name_to_idx_kept["base_link"]
                        kps_scene[i_base : i_base + 1] = apply_transform(
                            kps_scene[i_base : i_base + 1], T
                        )

                    # project regular keypoints
                    rc_by_name: Dict[str, Tuple[int, int]] = {}
                    vis_by_name: Dict[str, int] = {}

                    if kps_scene.size:
                        az_kp, el_kp = angles_from_xyz(kps_scene)
                        for jj, nm in enumerate(names_kept):
                            elv = float(el_kp[jj])
                            azv = float(az_kp[jj])
                            r = row_from_elevation(elv, el_per_row_calib, H)
                            c = col_from_azimuth_global(azv, az_per_col_calib, W)

                            # bounds check
                            if r < 0 or r >= H or c < 0 or c >= W:
                                vis_by_name[nm] = 0
                                continue

                            r_int = int(r)
                            c_int = int(c)

                            # base_link at top row → invisible
                            if nm == "base_link" and r_int == 0:
                                vis_by_name[nm] = 0
                                continue

                            # Ray-tracing visibility
                            if RAY_VISIBILITY_CHECK:
                                R_hit = get_min_depth(
                                    range_img,
                                    valid_range,
                                    r_int,
                                    c_int,
                                    RAY_PATCH_RADIUS,
                                )
                                R_kp = float(np.linalg.norm(kps_scene[jj]))

                                if R_hit is not None and np.isfinite(R_kp):
                                    if R_kp > R_hit + RAY_TOL:
                                        vis_by_name[nm] = 0
                                        continue

                            rc_by_name[nm] = (r_int, c_int)
                            vis_by_name[nm] = 1

                    # Synthetic front_wheels_mid
                    if (mid_raw_3d is not None) and (aircraft_pts.size > 0):
                        mid_adj_3d, _ = adjust_midpoint_to_cluster(
                            mid_raw_3d,
                            aircraft_pts,
                            z_min_air,
                            base_radius=MID_BASE_RADIUS,
                            expand_radius=MID_EXPAND_RADIUS,
                            z_band=MID_Z_BAND,
                            min_points=MID_MIN_POINTS,
                        )

                        az_syn, el_syn = angles_from_xyz(mid_adj_3d.reshape(1, 3))
                        r_syn = row_from_elevation(float(el_syn[0]), el_per_row_calib, H)
                        c_syn = col_from_azimuth_global(float(az_syn[0]), az_per_col_calib, W)

                        if 0 <= r_syn < H and 0 <= c_syn < W:
                            r_syn_int = int(r_syn)
                            c_syn_int = int(c_syn)

                            if RAY_VISIBILITY_CHECK:
                                R_hit_syn = get_min_depth(
                                    range_img,
                                    valid_range,
                                    r_syn_int,
                                    c_syn_int,
                                    RAY_PATCH_RADIUS,
                                )
                                R_kp_syn = float(np.linalg.norm(mid_adj_3d))

                                if R_hit_syn is not None and np.isfinite(R_kp_syn):
                                    if R_kp_syn > R_hit_syn + RAY_TOL:
                                        vis_by_name[SYN_KP_NAME] = 0
                                    else:
                                        rc_by_name[SYN_KP_NAME] = (r_syn_int, c_syn_int)
                                        vis_by_name[SYN_KP_NAME] = 1
                                else:
                                    rc_by_name[SYN_KP_NAME] = (r_syn_int, c_syn_int)
                                    vis_by_name[SYN_KP_NAME] = 1
                            else:
                                vis_by_name[SYN_KP_NAME] = 0
                    else:
                        vis_by_name[SYN_KP_NAME] = 0

                    # save image
                    img_path = (
                        Path(out_dir) / "images" / split_name / f"{unique_scene}.png"
                    )
                    imageio.imwrite(str(img_path), img, compress_level=1)

                    # visualizations
                    if MAKE_VIZ:
                        vis_img = img.copy()
                        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        for kp_name, (r0, c0) in rc_by_name.items():
                            if vis_by_name.get(kp_name, 0) <= 0:
                                continue
                            cv2.circle(
                                vis_img,
                                (int(c0), int(r0)),
                                3,
                                (0, 0, 255),
                                -1,
                                lineType=cv2.LINE_AA,
                            )
                            cv2.putText(
                                vis_img,
                                kp_name,
                                (int(c0) + 3, int(r0) - 3),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.35,
                                (255, 255, 255),
                                1,
                                cv2.LINE_AA,
                            )

                        cv2.putText(
                            vis_img,
                            f"{split_name}/{unique_scene}",
                            (5, 15),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )

                        vis_path = vis_root / split_name / f"{unique_scene}.png"
                        imageio.imwrite(str(vis_path), vis_img, compress_level=1)

                    # YOLO label
                    parts = [
                        "0",
                        f"{cx:.6f}",
                        f"{cy:.6f}",
                        f"{bw:.6f}",
                        f"{bh:.6f}",
                    ]
                    for kp in KP_ORDER:
                        if kp in rc_by_name and vis_by_name.get(kp, 0) > 0:
                            r0, c0 = rc_by_name[kp]
                            xn, yn = rc_to_xy_norm(r0, c0, H, W)
                            parts += [f"{xn:.6f}", f"{yn:.6f}", "1"]
                        else:
                            parts += ["0.000000", "0.000000", "0"]

                    (Path(out_dir) / "labels" / split_name / f"{unique_scene}.txt").write_text(
                        " ".join(parts) + "\n"
                    )

                    file_valid_scenes += 1
                    total_valid_scenes += 1

        except Exception as e:
            print(f"[ERROR] {Path(h5p).name}: {e}")

        if file_valid_scenes > 0:
            files_with_valid_scenes += 1
            print(f"  -> valid scenes in this file: {file_valid_scenes}")
        else:
            print("  -> no valid scenes exported from this file")

    print(f"\n[summary] Total valid scenes exported: {total_valid_scenes}")
    print(
        f"[summary] H5 files with >= 1 valid scene: {files_with_valid_scenes}/{total_files}"
    )

    # YAML
    yaml_text = (
        f"""# YOLO pose dataset — unified from multiple H5 files
path: {Path(out_dir).resolve()}
train: images/train
val: images/val
test: images/test

names: ["aircraft"]

kpt_shape: [{len(KP_ORDER)}, 3]
keypoints:
"""
        + "".join([f"  - {n}\n" for n in KP_ORDER])
    )
    (Path(out_dir) / "aircraft_pose.yaml").write_text(yaml_text)

    # diagnostics
    (Path(out_dir) / "export_config.json").write_text(
        json.dumps(
            {
                "source": source,
                "out_dir": out_dir,
                "split": split,
                "kp_order": KP_ORDER,
                "kpt_bbox_margin_px": KPT_BBOX_MARGIN_PX,
                "roll_wide_bbox": ROLL_WIDE_BBOX,
                "roll_wide_bbox_frac": ROLL_WIDE_BBOX_FRAC,
                "roll_wide_bbox_cols": ROLL_WIDE_BBOX_COLS,
                "max_h5_files": max_h5_files,
                "total_valid_scenes": total_valid_scenes,
                "files_with_valid_scenes": files_with_valid_scenes,
                "total_h5_files_seen": total_files,
                "visibility_definition": (
                    "1 = ray-visible (in image bounds, not base_link on row 0, and not clearly behind aircraft surface "
                    "in a local patch); 0 = not visible or occluded. Scenes with bbox_width / image_width > 0.6 are skipped."
                ),
                "ray_visibility_check": RAY_VISIBILITY_CHECK,
                "ray_tolerance_m": RAY_TOL,
                "ray_patch_radius": RAY_PATCH_RADIUS,
                "mid_base_radius": MID_BASE_RADIUS,
                "mid_expand_radius": MID_EXPAND_RADIUS,
                "mid_z_band": MID_Z_BAND,
                "mid_min_points": MID_MIN_POINTS,
            },
            indent=2,
        )
    )

    print("\n✓ Dataset exported to", Path(out_dir).resolve())
    print("  -> YAML:", (Path(out_dir) / "aircraft_pose.yaml").resolve())
    if MAKE_VIZ:
        print("  -> Visualizations under:", (Path(out_dir) / "vis").resolve())


def parse_arguments():
    """CLI overrides for SOURCE / OUT_DIR / SPLIT / MAX_H5_FILES."""
    parser = argparse.ArgumentParser(
        description="Create aircraft YOLO-pose dataset from HDF5 LiDAR files."
    )

    parser.add_argument(
        "--source",
        type=str,
        default=SOURCE,
        help=f"Input H5 directory or gs:// path (default: {SOURCE})",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=OUT_DIR,
        help=f"Output directory (default: {OUT_DIR})",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=SPLIT[0],
        help="Train split ratio (default from config_dataset.py)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=SPLIT[1],
        help="Validation split ratio",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=SPLIT[2],
        help="Test split ratio",
    )
    parser.add_argument(
        "--max-h5",
        type=int,
        default=MAX_H5_FILES,
        help="Limit number of H5 files (default from config_dataset.py)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    split_tuple = (args.train_ratio, args.val_ratio, args.test_ratio)

    create_dataset(
        source=args.source,
        out_dir=args.out,
        split=split_tuple,
        max_h5_files=args.max_h5,
    )
