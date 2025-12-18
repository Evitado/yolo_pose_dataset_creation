#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export VISUALIZED images only (bbox + keypoints) grouped by bag (H5 file).

Output:
  out_dir/by_bag/<bag_stem>/
      scene_000001_vis.png
      scene_000002_vis.png
      ...

NO raw images, NO labels, NO per-scene folders.

Works with SOURCE being local dir or gs:// prefix (via your io_helpers).
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import h5py
import imageio.v2 as imageio
import cv2

from config_dataset import (
    SOURCE,
    OUT_DIR,
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
    ROLL_WIDE_BBOX,
    ROLL_WIDE_BBOX_FRAC,
    ROLL_WIDE_BBOX_COLS,
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
    get_min_depth,
    adjust_midpoint_to_cluster,
    angles_from_xyz,
    row_from_elevation,
    col_from_azimuth_global,
    apply_transform,
    find_alias,
    _fill_nans,
)


def find_best_azimuth_roll(mask2d: np.ndarray) -> int:
    H, W = mask2d.shape
    col_has = mask2d.any(axis=0)
    empty = ~col_has
    if not np.any(empty):
        return 0

    best_len = 0
    best_start = 0
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

    seam_col = (best_start + best_len // 2) % W
    shift = -int(seam_col)
    return 0 if (shift % W == 0) else shift


def export_vis_only_by_bag(
    source: str,
    out_dir: str,
    max_h5_files: Optional[int] = None,
) -> None:
    np.random.seed(RANDOM_SEED)

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    by_bag = out_root / "by_bag"
    by_bag.mkdir(parents=True, exist_ok=True)

    print("[list] Searching for .h5 files…")
    h5_paths = list_h5_paths(source)
    if not h5_paths:
        raise RuntimeError(f"No .h5 files found under: {source}")
    print(f"[list] Found {len(h5_paths)} HDF5 files total.")

    if max_h5_files is not None and max_h5_files > 0 and len(h5_paths) > max_h5_files:
        h5_paths = h5_paths[:max_h5_files]
        print(f"[list] Using only first {len(h5_paths)} H5 files (max_h5_files={max_h5_files})")

    print("\n--- Phase 1: build unified keypoint order (for consistent drawing) ---")
    KP_ORDER: List[str] = []
    seen = set()

    for i, h5p in enumerate(h5_paths, 1):
        print(f"[{i}/{len(h5_paths)}] {Path(h5p).name}")
        try:
            with open_h5_any(h5p) as f:
                for _, g in f.items():
                    if isinstance(g, h5py.Group) and "keypoints" in g:
                        raw = g["keypoints"]["names"][()]
                        names = [
                            n.decode("utf-8") if isinstance(n, (bytes, bytearray)) else str(n)
                            for n in raw
                        ]
                        for n in names:
                            if n in REMOVE_KP_SET:
                                continue
                            if n not in seen:
                                KP_ORDER.append(n)
                                seen.add(n)
        except Exception as e:
            print(f"  [WARN] skipping {Path(h5p).name}: {e}")

    if SYN_KP_NAME not in KP_ORDER:
        KP_ORDER.append(SYN_KP_NAME)

    print(f"[kps] KP_ORDER ({len(KP_ORDER)}): {', '.join(KP_ORDER)}")

    print("\n--- Phase 2: export VIS images only (bbox + keypoints) ---")
    total_exported = 0

    for fi, h5p in enumerate(h5_paths, 1):
        bag_stem = Path(h5p).stem
        bag_dir = by_bag / bag_stem
        bag_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{fi}/{len(h5_paths)}] {Path(h5p).name} -> {bag_dir}")

        try:
            with open_h5_any(h5p) as f:
                H = int(f.attrs["height"])
                W = int(f.attrs["width"])

                scene_keys = sorted([k for k in f.keys() if k.startswith("scene_")])

                for scene_name in scene_keys:
                    grp = f[scene_name]
                    if "points" not in grp or "keypoints" not in grp:
                        continue

                    ds = grp["points"]
                    flat = ds[()]  # (H*W, C)

                    cols_raw = ds.attrs.get("columns", None)
                    cols = [
                        c.decode("utf-8") if isinstance(c, (bytes, bytearray)) else str(c)
                        for c in (cols_raw if cols_raw is not None else [])
                    ]
                    if not cols:
                        cols = ["x","y","z","range","intensity","reflectivity","ambient","is_ground","is_aircraft"]

                    if not {"x", "y", "z"}.issubset(set(cols)):
                        continue
                    if "is_aircraft" not in cols:
                        continue

                    ix, iy, iz = cols.index("x"), cols.index("y"), cols.index("z")
                    mask2d = flat[:, cols.index("is_aircraft")].astype(np.uint8).reshape(H, W).astype(bool)

                    # image
                    rgb = build_rgb_from_cols(flat, cols, H, W)
                    if rgb is None:
                        gray = (mask2d.astype(np.uint8) * 255)
                        rgb = np.dstack([gray, gray, gray])
                    img = rgb

                    if DRAW_ON_OVERLAY:
                        overlay = img.copy()
                        red = np.zeros_like(overlay)
                        red[..., 0] = 255
                        alpha = 0.5
                        overlay[mask2d] = (alpha * red[mask2d] + (1 - alpha) * overlay[mask2d]).astype(np.uint8)
                        img = overlay

                    if APPLY_MEDIAN_FILTER:
                        if img.dtype != np.uint8:
                            img = img.astype(np.uint8)
                        img = cv2.medianBlur(img, MEDIAN_KSIZE)

                    # xyz
                    xyz = np.stack([flat[:, ix], flat[:, iy], flat[:, iz]], axis=1).astype(np.float64)
                    xyz_hw3 = xyz.reshape(H, W, 3)

                    # bbox
                    bb = bbox_from_mask(mask2d)
                    if bb is None:
                        continue
                    x1, y1, x2, y2 = bb
                    bbox_w = (x2 - x1 + 1)
                    bbox_frac = bbox_w / float(W)

                    # roll if wide
                    if ROLL_WIDE_BBOX and bbox_frac > ROLL_WIDE_BBOX_FRAC and W > 1:
                        shift = find_best_azimuth_roll(mask2d)
                        if shift == 0:
                            shift = ROLL_WIDE_BBOX_COLS % W

                        if shift != 0:
                            img = np.roll(img, shift=shift, axis=1)
                            mask2d = np.roll(mask2d, shift=shift, axis=1)
                            xyz_hw3 = np.roll(xyz_hw3, shift=shift, axis=1)

                            bb2 = bbox_from_mask(mask2d)
                            if bb2 is None:
                                continue
                            x1, y1, x2, y2 = bb2
                            bbox_w = (x2 - x1 + 1)
                            bbox_frac = bbox_w / float(W)

                        if bbox_frac > 0.6:
                            continue

                    aircraft_pts = xyz_hw3[mask2d]
                    z_min_air = float(np.min(aircraft_pts[:, 2])) if aircraft_pts.size > 0 else 0.0

                    # range image for ray visibility
                    range_img = np.linalg.norm(xyz_hw3, axis=2)
                    valid_range = np.isfinite(range_img) & mask2d

                    # calibration
                    valid = np.all(np.isfinite(xyz_hw3), axis=2)
                    el_all = np.full((H, W), np.nan, dtype=np.float64)
                    az_all = np.full((H, W), np.nan, dtype=np.float64)

                    xv = xyz_hw3[..., 0][valid]
                    yv = xyz_hw3[..., 1][valid]
                    zv = xyz_hw3[..., 2][valid]
                    el_all[valid] = np.arctan2(zv, np.sqrt(xv * xv + yv * yv))
                    az_all[valid] = np.arctan2(yv, xv)

                    el_per_row_calib = _fill_nans(np.nanmedian(el_all, axis=1))
                    sin_c = np.nanmean(np.sin(az_all), axis=0)
                    cos_c = np.nanmean(np.cos(az_all), axis=0)
                    az_per_col_calib = _fill_nans(np.arctan2(sin_c, cos_c))

                    # keypoints
                    kp_grp = grp["keypoints"]
                    kps_model = np.asarray(kp_grp["xyz"][()], dtype=np.float64)

                    raw_names = kp_grp.get("names", None)
                    scene_names_list = [
                        n.decode("utf-8") if isinstance(n, (bytes, bytearray)) else str(n)
                        for n in (raw_names[()] if raw_names is not None else [])
                    ]
                    if kps_model.ndim != 2 or kps_model.shape[1] != 3 or kps_model.shape[0] == 0:
                        continue

                    ok_rows = np.all(np.isfinite(kps_model), axis=1)
                    kps_model = kps_model[ok_rows]
                    if scene_names_list:
                        scene_names_list = [scene_names_list[i] for i, t in enumerate(ok_rows) if t]
                    else:
                        scene_names_list = [f"k{i}" for i in range(kps_model.shape[0])]

                    name_to_idx_full = {n: i for i, n in enumerate(scene_names_list)}

                    # midpoint raw
                    mid_raw_3d = None
                    nm_fr = find_alias(scene_names_list, FRONT_RIGHT_ALIASES)
                    nm_fl = find_alias(scene_names_list, FRONT_LEFT_ALIASES)
                    if (nm_fr in name_to_idx_full) and (nm_fl in name_to_idx_full):
                        p_fr = kps_model[name_to_idx_full[nm_fr]]
                        p_fl = kps_model[name_to_idx_full[nm_fl]]
                        mid_raw_3d = 0.5 * (p_fr + p_fl)

                    # remove unwanted
                    keep_mask = [n not in REMOVE_KP_SET for n in scene_names_list]
                    names_kept = [n for n, keep in zip(scene_names_list, keep_mask) if keep]
                    kps_kept = kps_model[keep_mask]
                    name_to_idx_kept = {n: i for i, n in enumerate(names_kept)}

                    # transform base_link only
                    T = None
                    if USE_TF_MATRIX and "metadata" in grp and "tf_matrix" in grp["metadata"]:
                        T_ = np.asarray(grp["metadata"]["tf_matrix"][()], dtype=np.float64)
                        if T_.shape == (4, 4) and np.all(np.isfinite(T_)):
                            T = T_ if not APPLY_Z_FLIP else (T_ @ np.diag([1.0, 1.0, -1.0, 1.0]))

                    kps_scene = kps_kept.copy()
                    if ("base_link" in name_to_idx_kept) and (T is not None):
                        i_base = name_to_idx_kept["base_link"]
                        kps_scene[i_base:i_base+1] = apply_transform(kps_scene[i_base:i_base+1], T)

                    # project + visibility
                    rc_by_name: Dict[str, Tuple[int, int]] = {}
                    vis_by_name: Dict[str, int] = {}

                    if kps_scene.size:
                        az_kp, el_kp = angles_from_xyz(kps_scene)
                        for jj, nm in enumerate(names_kept):
                            r = row_from_elevation(float(el_kp[jj]), el_per_row_calib, H)
                            c = col_from_azimuth_global(float(az_kp[jj]), az_per_col_calib, W)

                            if r < 0 or r >= H or c < 0 or c >= W:
                                vis_by_name[nm] = 0
                                continue

                            r_int, c_int = int(r), int(c)

                            if nm == "base_link" and r_int == 0:
                                vis_by_name[nm] = 0
                                continue

                            if RAY_VISIBILITY_CHECK:
                                R_hit = get_min_depth(range_img, valid_range, r_int, c_int, RAY_PATCH_RADIUS)
                                R_kp = float(np.linalg.norm(kps_scene[jj]))
                                if R_hit is not None and np.isfinite(R_kp) and (R_kp > R_hit + RAY_TOL):
                                    vis_by_name[nm] = 0
                                    continue

                            rc_by_name[nm] = (r_int, c_int)
                            vis_by_name[nm] = 1

                    # synthetic mid
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
                            r_syn_int, c_syn_int = int(r_syn), int(c_syn)
                            if RAY_VISIBILITY_CHECK:
                                R_hit_syn = get_min_depth(range_img, valid_range, r_syn_int, c_syn_int, RAY_PATCH_RADIUS)
                                R_kp_syn = float(np.linalg.norm(mid_adj_3d))
                                if R_hit_syn is not None and np.isfinite(R_kp_syn) and (R_kp_syn > R_hit_syn + RAY_TOL):
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

                    # --- draw VIS image ---
                    vis_img = img.copy()

                    # bbox
                    cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # keypoints (only visible)
                    for kp_name in KP_ORDER:
                        if kp_name not in rc_by_name:
                            continue
                        if vis_by_name.get(kp_name, 0) <= 0:
                            continue
                        r0, c0 = rc_by_name[kp_name]
                        cv2.circle(vis_img, (int(c0), int(r0)), 3, (0, 0, 255), -1, lineType=cv2.LINE_AA)
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

                    # save
                    stem = scene_name  # e.g. scene_000123
                    out_path = bag_dir / f"{stem}_vis.png"
                    imageio.imwrite(str(out_path), vis_img, compress_level=1)

                    total_exported += 1

        except Exception as e:
            print(f"[WARN] {Path(h5p).name}: {e}")

    print(f"\n✓ Exported {total_exported} visualized scenes")
    print("  -> Output:", by_bag.resolve())


def parse_arguments():
    p = argparse.ArgumentParser("Export VIS images only (bbox + keypoints), grouped by bag")
    p.add_argument("--source", type=str, default=SOURCE)
    p.add_argument("--out", type=str, default=OUT_DIR)
    p.add_argument("--max-h5", type=int, default=MAX_H5_FILES)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    export_vis_only_by_bag(
        source=args.source,
        out_dir=args.out,
        max_h5_files=args.max_h5,
    )
