#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NO-ARGS YOLO-pose inference on point cloud / H5 range-image + optional image input.

Modes:
  - H5: loads xyz_hw3 directly from H5 range-image grid
  - PCD/PLY: projects point cloud to spherical panorama (H x W)
  - IMAGE_ONLY: runs YOLO on an image (2D only, no backprojection)

Optional:
  - IMG_OVERRIDE: run YOLO on a provided image, while still using xyz_hw3 from H5/PCD.
    (Requires same H,W and same alignment/roll.)

Outputs:
  - saves input + overlay PNGs
  - Open3D visualization: base point cloud + bbox point subsets + OBB lines + keypoint spheres

Deps:
  pip install ultralytics open3d opencv-python h5py imageio numpy
"""

import math
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import cv2
import imageio.v2 as imageio

import open3d as o3d
from ultralytics import YOLO

try:
    import h5py
except Exception:
    h5py = None


# ============================================================
# CONFIG (edit these, then run the file)
# ============================================================

WEIGHTS = "/home/femi/yolo_pose_dataset_creation/runs/pose/aircraft_pose_new_clear_img8/weights/best.pt"
OUT_DIR = "./infer_out_noargs2"

# Choose ONE main source for xyz mapping:
H5_PATH: Optional[str]  = "/home/femi/prof/h5_files_v2/movement_a350_900__2025-08-29T04-49-09.h5"     # e.g. "/path/bag_001.h5"
H5_SCENE: Optional[str] = None     # e.g. "scene_000012" (None = auto)
PCD_PATH: Optional[str] = None     # e.g. "/path/cloud.pcd" or ".ply"

# Optional: provide an image to run YOLO on
# - If you ALSO set H5_PATH or PCD_PATH, it will use that image for YOLO and xyz_hw3 for backprojection
# - If you set ONLY IMG_PATH (and no H5/PCD), it will do 2D-only inference
IMG_PATH: Optional[str] = None     # e.g. "./some_range_image.png"

# Panorama size if using PCD/PLY projection
PAN_H = 128
PAN_W = 1024
EL_MIN_RAD: Optional[float] = None
EL_MAX_RAD: Optional[float] = None

# YOLO thresholds
CONF = 0.25
IOU  = 0.45
MAX_DET = 10

# Keypoint handling
KP_CONF = 0.25
PATCH_RADIUS = 3  # if pixel is empty/NaN, search neighbor patch for a valid xyz

# If you have your aircraft_pose.yaml, keypoint names will be drawn on overlay
YAML_KP_NAMES: Optional[str] = None  # e.g. "./aircraft_pose.yaml"

# H5 seam-roll options (only available if H5 has is_aircraft mask)
DO_ROLL_IF_WIDE = True
ROLL_WIDE_BBOX_FRAC = 0.90
ROLL_FALLBACK_COLS = 512  # used if "best gap" roll fails
APPLY_MEDIAN_FILTER = True
MEDIAN_KSIZE = 3

SHOW_3D = True
SAVE_PLY = False


# ============================================================
# Helpers
# ============================================================

def autoscale_01(x: np.ndarray, nan_fill: float = 0.0) -> np.ndarray:
    x = np.nan_to_num(x, nan=nan_fill)
    vmin, vmax = np.percentile(x, (1, 99))
    denom = (vmax - vmin) if vmax > vmin else 1.0
    return np.clip((x - vmin) / (denom + 1e-12), 0.0, 1.0)

def build_rgb_from_cols(flat: np.ndarray, cols: List[str], H: int, W: int) -> np.ndarray:
    idx = {c: i for i, c in enumerate(cols)}
    if all(k in idx for k in ("reflectivity", "range", "intensity")):
        refl  = flat[:, idx["reflectivity"]].astype(np.float64)
        rng   = flat[:, idx["range"]].astype(np.float64)
        inten = flat[:, idx["intensity"]].astype(np.float64)

        rng_safe = np.clip(rng, 1e-3, None)
        log_range = np.log(rng_safe)

        r = (autoscale_01(refl) * 255).astype(np.uint8).reshape(H, W)
        g = (autoscale_01(inten) * 255).astype(np.uint8).reshape(H, W)
        b = ((1.0 - autoscale_01(log_range)) * 255).astype(np.uint8).reshape(H, W)
        return np.stack([r, g, b], axis=-1)

    # fallback
    return np.zeros((H, W, 3), dtype=np.uint8)

def angles_from_xyz(xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    az = np.arctan2(y, x)
    el = np.arctan2(z, np.sqrt(x * x + y * y))
    return az, el

def bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

def find_best_azimuth_roll(mask2d: np.ndarray) -> int:
    H, W = mask2d.shape
    col_has_aircraft = mask2d.any(axis=0)
    empty = ~col_has_aircraft
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
    if shift % W == 0:
        return 0
    return shift

def nearest_valid_xyz_in_patch(xyz_hw3: np.ndarray, r: int, c: int, patch_radius: int) -> Optional[np.ndarray]:
    H, W, _ = xyz_hw3.shape
    r0 = max(0, r - patch_radius)
    r1 = min(H, r + patch_radius + 1)
    c0 = max(0, c - patch_radius)
    c1 = min(W, c + patch_radius + 1)

    patch = xyz_hw3[r0:r1, c0:c1, :]
    valid = np.all(np.isfinite(patch), axis=2)
    if not np.any(valid):
        return None

    ys, xs = np.nonzero(valid)
    rr = ys + r0
    cc = xs + c0
    d2 = (rr - r) ** 2 + (cc - c) ** 2
    k = int(np.argmin(d2))
    return xyz_hw3[int(rr[k]), int(cc[k]), :].copy()

def to_numpy(x):
    try:
        return x.detach().cpu().numpy()
    except Exception:
        return np.asarray(x)

def load_kp_names_from_yaml(yaml_path: Optional[str]) -> Optional[List[str]]:
    if yaml_path is None:
        return None
    p = Path(yaml_path)
    if not p.exists():
        return None
    lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    names = []
    in_kp = False
    for ln in lines:
        s = ln.strip()
        if s.startswith("keypoints:"):
            in_kp = True
            continue
        if in_kp:
            if s.startswith("- "):
                names.append(s[2:].strip())
            elif s == "" or s.startswith("#"):
                continue
            elif ":" in s and not s.startswith("-"):
                break
    return names if names else None

def make_sphere(center: np.ndarray, radius: float = 0.25) -> o3d.geometry.TriangleMesh:
    sph = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=12)
    sph.translate(center.astype(np.float64))
    sph.compute_vertex_normals()
    return sph


# ============================================================
# Load / project
# ============================================================

def load_image_rgb(path: str) -> np.ndarray:
    img = imageio.imread(path)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def load_h5_range_image(h5_path: str, scene: Optional[str]) -> Tuple[np.ndarray, np.ndarray]:
    if h5py is None:
        raise RuntimeError("h5py missing. Install: pip install h5py")

    with h5py.File(h5_path, "r") as f:
        H = int(f.attrs["height"])
        W = int(f.attrs["width"])

        if scene is None:
            scene = None
            for s, g in f.items():
                if isinstance(g, h5py.Group) and "points" in g:
                    scene = s
                    break
            if scene is None:
                raise RuntimeError("No scene found in H5.")

        grp = f[scene]
        ds = grp["points"]
        flat = ds[()]

        cols_raw = ds.attrs.get("columns", None)
        cols = [c.decode("utf-8") if isinstance(c, (bytes, bytearray)) else str(c)
                for c in (cols_raw if cols_raw is not None else [])]
        if not cols:
            cols = ["x", "y", "z", "range", "intensity", "reflectivity", "is_aircraft"]

        if not {"x", "y", "z"}.issubset(set(cols)):
            raise RuntimeError("Missing x/y/z in H5 columns.")

        ix, iy, iz = cols.index("x"), cols.index("y"), cols.index("z")
        xyz = np.stack([flat[:, ix], flat[:, iy], flat[:, iz]], axis=1).astype(np.float64)
        xyz_hw3 = xyz.reshape(H, W, 3)
        finite = np.all(np.isfinite(xyz_hw3), axis=2)
        xyz_hw3[~finite] = np.nan

        img = build_rgb_from_cols(flat, cols, H, W)

        # Optional roll (if mask exists)
        if DO_ROLL_IF_WIDE and ("is_aircraft" in cols):
            mask = flat[:, cols.index("is_aircraft")].astype(np.uint8).reshape(H, W).astype(bool)
            bb = bbox_from_mask(mask)
            if bb is not None:
                x1, y1, x2, y2 = bb
                bbox_w = (x2 - x1 + 1)
                frac = bbox_w / float(W)
                if frac > ROLL_WIDE_BBOX_FRAC:
                    shift = find_best_azimuth_roll(mask)
                    if shift == 0:
                        shift = int(ROLL_FALLBACK_COLS) % W
                    if shift != 0:
                        print(f"[roll] wide mask bbox frac={frac:.3f} -> shift={shift}")
                        img = np.roll(img, shift=shift, axis=1)
                        xyz_hw3 = np.roll(xyz_hw3, shift=shift, axis=1)

        if APPLY_MEDIAN_FILTER:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img_bgr = cv2.medianBlur(img_bgr, MEDIAN_KSIZE)
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    return img, xyz_hw3

def project_pcd_to_panorama(pcd_path: str, H: int, W: int,
                            el_min: Optional[float], el_max: Optional[float]) -> Tuple[np.ndarray, np.ndarray]:
    pcd = o3d.io.read_point_cloud(pcd_path)
    xyz = np.asarray(pcd.points, dtype=np.float64)
    if xyz.size == 0:
        raise RuntimeError("Empty point cloud.")

    az, el = angles_from_xyz(xyz)
    rng = np.linalg.norm(xyz, axis=1)

    if el_min is None or el_max is None:
        lo, hi = np.percentile(el[np.isfinite(el)], [1, 99])
        el_min = float(lo) if el_min is None else float(el_min)
        el_max = float(hi) if el_max is None else float(el_max)
        if el_max <= el_min + 1e-6:
            el_min, el_max = float(np.min(el)), float(np.max(el))

    c = ((az + math.pi) / (2 * math.pi) * W)
    r = ((el - el_min) / (el_max - el_min + 1e-12) * H)

    cc = np.clip(c.astype(np.int64), 0, W - 1)
    rr = np.clip(r.astype(np.int64), 0, H - 1)

    xyz_hw3 = np.full((H, W, 3), np.nan, dtype=np.float64)
    rimg = np.full((H, W), np.inf, dtype=np.float64)

    for i in range(xyz.shape[0]):
        ri = int(rr[i]); ci = int(cc[i])
        if rng[i] < rimg[ri, ci]:
            rimg[ri, ci] = float(rng[i])
            xyz_hw3[ri, ci, :] = xyz[i, :]

    depth = rimg.copy()
    depth[~np.isfinite(depth)] = np.nan
    g = (autoscale_01(depth, nan_fill=np.nanmax(depth[np.isfinite(depth)]) if np.any(np.isfinite(depth)) else 0.0) * 255).astype(np.uint8)
    img = np.dstack([g, g, g])

    return img, xyz_hw3


# ============================================================
# Main
# ============================================================

def run():
    out = Path(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    kp_names = load_kp_names_from_yaml(YAML_KP_NAMES)

    # Decide xyz mapping source
    have_xyz = (H5_PATH is not None) or (PCD_PATH is not None)

    base_img_rgb = None
    xyz_hw3 = None
    stem = "infer"

    if H5_PATH is not None:
        base_img_rgb, xyz_hw3 = load_h5_range_image(H5_PATH, H5_SCENE)
        stem = f"{Path(H5_PATH).stem}__{H5_SCENE or 'auto'}"
    elif PCD_PATH is not None:
        base_img_rgb, xyz_hw3 = project_pcd_to_panorama(PCD_PATH, PAN_H, PAN_W, EL_MIN_RAD, EL_MAX_RAD)
        stem = Path(PCD_PATH).stem

    # If image is provided, it becomes YOLO input (override)
    if IMG_PATH is not None:
        img_in_rgb = load_image_rgb(IMG_PATH)
        if have_xyz:
            H0, W0 = base_img_rgb.shape[:2]
            H1, W1 = img_in_rgb.shape[:2]
            if (H0 != H1) or (W0 != W1):
                raise RuntimeError(
                    f"IMG_PATH size {H1}x{W1} does not match xyz mapping image {H0}x{W0}. "
                    f"Do NOT resize; the pixel->3D mapping would break."
                )
            print("[img] Using IMG_PATH as YOLO input (with xyz backprojection).")
        else:
            print("[img] IMAGE_ONLY mode (2D only, no backprojection).")
        yolo_img_rgb = img_in_rgb
        stem = f"{stem}__img"
    else:
        if base_img_rgb is None:
            raise RuntimeError("No input set. Provide H5_PATH or PCD_PATH or IMG_PATH.")
        yolo_img_rgb = base_img_rgb

    # Save input
    imageio.imwrite(str(out / f"{stem}__input.png"), yolo_img_rgb, compress_level=1)

    # Run YOLO
    model = YOLO(WEIGHTS)
    r0 = model.predict(
        source=yolo_img_rgb,
        conf=CONF,
        iou=IOU,
        max_det=MAX_DET,
        verbose=False,
    )[0]

    # Prepare overlay (draw with cv2 on BGR, then convert back to RGB)
    overlay_bgr = cv2.cvtColor(yolo_img_rgb, cv2.COLOR_RGB2BGR)

    if r0.boxes is None or len(r0.boxes) == 0:
        print("[yolo] no detections")
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
        imageio.imwrite(str(out / f"{stem}__overlay.png"), overlay_rgb, compress_level=1)
        return

    boxes_xyxy = to_numpy(r0.boxes.xyxy)
    box_conf   = to_numpy(r0.boxes.conf) if getattr(r0.boxes, "conf", None) is not None else None

    kp_xy = None
    kp_conf = None
    if getattr(r0, "keypoints", None) is not None and r0.keypoints is not None:
        kp_xy = to_numpy(r0.keypoints.xy)
        if getattr(r0.keypoints, "conf", None) is not None and r0.keypoints.conf is not None:
            kp_conf = to_numpy(r0.keypoints.conf)

    H, W = yolo_img_rgb.shape[:2]

    # Build 3D scene if possible
    geoms = []
    if have_xyz and xyz_hw3 is not None:
        valid_all = np.all(np.isfinite(xyz_hw3), axis=2)
        pts_all = xyz_hw3[valid_all].reshape(-1, 3)
        pcd_all = o3d.geometry.PointCloud()
        pcd_all.points = o3d.utility.Vector3dVector(pts_all)
        pcd_all.colors = o3d.utility.Vector3dVector(np.tile(np.array([[0.7, 0.7, 0.7]]), (pts_all.shape[0], 1)))
        geoms.append(pcd_all)
        geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0))

    detN = boxes_xyxy.shape[0]
    print(f"[yolo] detections: {detN}")

    for i in range(detN):
        x1, y1, x2, y2 = boxes_xyxy[i]
        conf_i = float(box_conf[i]) if box_conf is not None else 0.0

        x1i = int(np.clip(x1, 0, W - 1))
        x2i = int(np.clip(x2, 0, W - 1))
        y1i = int(np.clip(y1, 0, H - 1))
        y2i = int(np.clip(y2, 0, H - 1))

        cv2.rectangle(overlay_bgr, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
        cv2.putText(
            overlay_bgr,
            f"det{i} {conf_i:.2f}",
            (x1i, max(0, y1i - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Backproject bbox to 3D
        if have_xyz and xyz_hw3 is not None:
            valid_all = np.all(np.isfinite(xyz_hw3), axis=2)
            bbox_mask = np.zeros((H, W), dtype=bool)
            bbox_mask[y1i:y2i + 1, x1i:x2i + 1] = True
            sel = bbox_mask & valid_all
            pts_bbox = xyz_hw3[sel].reshape(-1, 3)

            if pts_bbox.shape[0] >= 10:
                pcd_bbox = o3d.geometry.PointCloud()
                pcd_bbox.points = o3d.utility.Vector3dVector(pts_bbox)
                pcd_bbox.colors = o3d.utility.Vector3dVector(np.tile(np.array([[1.0, 0.9, 0.2]]), (pts_bbox.shape[0], 1)))
                geoms.append(pcd_bbox)

                obb = pcd_bbox.get_oriented_bounding_box()
                ls = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
                ls.paint_uniform_color([0.1, 1.0, 0.1])
                geoms.append(ls)

        # Keypoints
        if kp_xy is not None and i < kp_xy.shape[0]:
            K = kp_xy.shape[1]
            for k in range(K):
                xk, yk = kp_xy[i, k]
                rk = int(np.clip(yk, 0, H - 1))
                ck = int(np.clip(xk, 0, W - 1))

                ok = True
                if kp_conf is not None:
                    ok = float(kp_conf[i, k]) >= KP_CONF
                if not ok:
                    continue

                cv2.circle(overlay_bgr, (ck, rk), 3, (0, 0, 255), -1, lineType=cv2.LINE_AA)
                if kp_names is not None and k < len(kp_names):
                    cv2.putText(
                        overlay_bgr,
                        kp_names[k],
                        (ck + 3, rk - 3),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

                # Backproject kp to 3D
                if have_xyz and xyz_hw3 is not None:
                    xyz_k = None
                    if np.all(np.isfinite(xyz_hw3[rk, ck, :])):
                        xyz_k = xyz_hw3[rk, ck, :].copy()
                    else:
                        xyz_k = nearest_valid_xyz_in_patch(xyz_hw3, rk, ck, PATCH_RADIUS)

                    if xyz_k is None:
                        continue

                    sph = make_sphere(xyz_k, radius=0.25)
                    sph.paint_uniform_color([1.0, 0.0, 0.0])
                    geoms.append(sph)

    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    imageio.imwrite(str(out / f"{stem}__overlay.png"), overlay_rgb, compress_level=1)
    print("[save] overlay:", out / f"{stem}__overlay.png")

    if have_xyz and SAVE_PLY and len(geoms) > 0 and isinstance(geoms[0], o3d.geometry.PointCloud):
        ply_path = out / f"{stem}__base_points.ply"
        o3d.io.write_point_cloud(str(ply_path), geoms[0])
        print("[save] ply:", ply_path)

    if SHOW_3D and have_xyz and len(geoms) > 0:
        o3d.visualization.draw_geometries(geoms, window_name="YOLO backproject", width=1280, height=720)


if __name__ == "__main__":
    run()
