#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Offline augmenter for YOLO pose dataset (aircraft range images).

- Works on exported dataset structure:
    root/
      images/train/*.png
      labels/train/*.txt

- For each image/label pair:
    * creates several azimuth-roll (horizontal circular shift) variants
    * creates several scale (zoom in/out) variants

Assumes label format (single line per file):
  class cx cy w h kp1x kp1y kp1v kp2x kp2y kp2v ...

All coords normalized to [0,1], visibility v in {0,1}.
Augmented samples are filtered so that:
  - all visible keypoints stay near the bbox (same logic as exporter)
  - bbox width <= 0.45 of image width (normalized w <= 0.45)

Additionally:
  - For each accepted augmented sample, a visualization image with bbox + keypoints
    is saved under: DATA_ROOT / "vis" / SPLIT / <stem>.png
"""

import os
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

# =========================
# CONFIG
# =========================
DATA_ROOT = Path("../aircraft_pose_all")  # adjust if needed
SPLIT = "train"  # usually augment only train ("train", "val", or "test")

# How many and how strong augmentations
ROLL_SHIFTS = [16, 32, 64, 128, 196, 256, 384, 512, 768]  # horizontal circular shifts (in pixels)
SCALE_FACTORS = [0.75, 0.85, 0.95, 1.05, 1.15, 1.25]      # <1 = zoom out, >1 = zoom in (around center)

# Name suffixes
ROLL_SUFFIX = "roll"
SCALE_SUFFIX = "scale"

# If True, skip images that already look augmented (contain suffix)
SKIP_ALREADY_AUG = True

# Same as exporter logic
KPT_BBOX_MARGIN_PX = 30      # allow keypoints to be this many pixels outside bbox
BBOX_MAX_FRAC = 0.45         # skip if bbox width > 45% of image width (normalized w > 0.45)

# Visualization for augmented samples
MAKE_VIZ = True  # set False if you don't want vis outputs for augmentations


# =========================
# YOLO label helpers
# =========================
def load_yolo_pose_label(label_path: Path) -> Tuple[int, float, float, float, float, np.ndarray]:
    """
    Load a single-line YOLO pose label file.

    Returns:
        cls_id: int
        cx, cy, w, h: floats (normalized 0..1)
        kpts: np.ndarray shape (K,3) with (x,y,v) normalized + visibility
    """
    txt = label_path.read_text().strip()
    if not txt:
        raise ValueError(f"Empty label file: {label_path}")
    parts = txt.split()
    cls_id = int(parts[0])
    cx = float(parts[1]); cy = float(parts[2])
    w = float(parts[3]);  h = float(parts[4])
    rest = np.array([float(x) for x in parts[5:]], dtype=np.float32)
    if rest.size % 3 != 0:
        raise ValueError(f"Keypoints length not multiple of 3 in {label_path}")
    kpts = rest.reshape(-1, 3)  # (K,3) x,y,v
    return cls_id, cx, cy, w, h, kpts


def save_yolo_pose_label(
    label_path: Path,
    cls_id: int,
    cx: float,
    cy: float,
    w: float,
    h: float,
    kpts: np.ndarray
):
    """
    Save YOLO pose label back to disk.
    """
    parts = [
        str(cls_id),
        f"{cx:.6f}", f"{cy:.6f}",
        f"{w:.6f}",  f"{h:.6f}",
    ]
    for x, y, v in kpts:
        parts.append(f"{x:.6f}")
        parts.append(f"{y:.6f}")
        parts.append(str(int(round(v))))
    label_path.write_text(" ".join(parts) + "\n")


# =========================
# BBOX / KPT CHECK HELPERS
# =========================
def kp_inside_or_near_bbox_px(x1, y1, x2, y2, x, y, margin_px: int) -> bool:
    return (
        x >= x1 - margin_px and x <= x2 + margin_px and
        y >= y1 - margin_px and y <= y2 + margin_px
    )


def all_kpts_ok_with_bbox(
    W: int,
    H: int,
    cx: float,
    cy: float,
    w: float,
    h: float,
    kpts: np.ndarray,
    margin_px: int = KPT_BBOX_MARGIN_PX,
) -> bool:
    """
    Rebuild bbox in pixel coords and check all visible keypoints (v>0)
    are inside or within margin_px of the bbox.
    """
    # bbox in pixels
    x1 = (cx - w / 2.0) * W
    y1 = (cy - h / 2.0) * H
    x2 = (cx + w / 2.0) * W
    y2 = (cy + h / 2.0) * H

    for x, y, v in kpts:
        if v <= 0.0:
            continue
        x_px = x * W
        y_px = y * H
        if not kp_inside_or_near_bbox_px(x1, y1, x2, y2, x_px, y_px, margin_px):
            return False
    return True


# =========================
# Visualization helper
# =========================
def save_vis_image(
    vis_dir: Path,
    stem: str,
    img: np.ndarray,
    cx: float,
    cy: float,
    w: float,
    h: float,
    kpts: np.ndarray,
):
    """
    Save a visualization image with bbox and keypoints drawn.

    - vis_dir / f"{stem}.png"
    """
    H, W = img.shape[:2]

    # bbox in px
    x1 = int((cx - w / 2.0) * W)
    y1 = int((cy - h / 2.0) * H)
    x2 = int((cx + w / 2.0) * W)
    y2 = int((cy + h / 2.0) * H)

    # clamp to image bounds
    x1 = max(0, min(W - 1, x1))
    x2 = max(0, min(W - 1, x2))
    y1 = max(0, min(H - 1, y1))
    y2 = max(0, min(H - 1, y2))

    vis_img = img.copy()

    # draw bbox (green)
    cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # draw keypoints (red), label with index
    for idx, (x, y, v) in enumerate(kpts):
        if v <= 0.0:
            continue
        x_px = int(x * W)
        y_px = int(y * H)
        if x_px < 0 or x_px >= W or y_px < 0 or y_px >= H:
            continue
        cv2.circle(vis_img, (x_px, y_px), 3, (0, 0, 255), -1, lineType=cv2.LINE_AA)
        cv2.putText(
            vis_img, str(idx),
            (x_px + 3, y_px - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35, (255, 255, 255), 1, cv2.LINE_AA
        )

    out_path = vis_dir / f"{stem}.png"
    cv2.imwrite(str(out_path), vis_img)


# =========================
# Augmentations
# =========================
def augment_roll(
    img: np.ndarray,
    cls_id: int,
    cx: float,
    cy: float,
    w: float,
    h: float,
    kpts: np.ndarray,
    shift_px: int
):
    """
    Horizontal circular roll by shift_px columns.

    - Image: np.roll along axis=1
    - bbox cx: shift in x
    - keypoints x: shift in x
    """
    H, W = img.shape[:2]

    # ---- image ----
    rolled_img = np.roll(img, shift=shift_px, axis=1)

    # ---- bbox center ----
    cx_px = cx * W
    cx_px_rolled = (cx_px + shift_px) % W
    cx_new = cx_px_rolled / W

    # ---- keypoints ----
    kpts_new = kpts.copy()
    for i in range(kpts_new.shape[0]):
        x, y, v = kpts_new[i]
        if v <= 0.0:
            continue
        x_px = x * W
        x_px_rolled = (x_px + shift_px) % W
        kpts_new[i, 0] = x_px_rolled / W
        # y unchanged

    return rolled_img, cx_new, cy, w, h, kpts_new


def augment_scale(
    img: np.ndarray,
    cls_id: int,
    cx: float,
    cy: float,
    w: float,
    h: float,
    kpts: np.ndarray,
    scale: float
):
    """
    Uniform scale around image center using cv2.warpAffine.

    - Image is scaled about the center with factor "scale" and kept at same size (H,W).
    - bbox + keypoints scaled around center accordingly.
    - Keypoints that go fully outside are marked invisible (v=0, x=y set to 0).
    """
    H, W = img.shape[:2]
    cx_img = W / 2.0
    cy_img = H / 2.0

    # ---- image: warpAffine with scaling around center ----
    M = cv2.getRotationMatrix2D((cx_img, cy_img), 0.0, scale)
    scaled_img = cv2.warpAffine(
        img, M, (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    # ---- bbox: scale around center ----
    cx_px = cx * W
    cy_px = cy * H
    w_px  = w * W
    h_px  = h * H

    # center scaled about image center
    cx_px_new = cx_img + scale * (cx_px - cx_img)
    cy_px_new = cy_img + scale * (cy_px - cy_img)
    # width/height scaled
    w_px_new  = w_px * scale
    h_px_new  = h_px * scale

    # convert back to normalized
    cx_new = cx_px_new / W
    cy_new = cy_px_new / H
    w_new  = w_px_new  / W
    h_new  = h_px_new  / H

    # clip bbox to [0,1]
    cx_new = float(np.clip(cx_new, 0.0, 1.0))
    cy_new = float(np.clip(cy_new, 0.0, 1.0))
    w_new  = float(np.clip(w_new,  1e-6, 1.0))
    h_new  = float(np.clip(h_new,  1e-6, 1.0))

    # ---- keypoints: scale around center ----
    kpts_new = kpts.copy()
    for i in range(kpts_new.shape[0]):
        x, y, v = kpts_new[i]
        if v <= 0.0:
            continue

        x_px = x * W
        y_px = y * H

        # scale about center
        x_px_new = cx_img + scale * (x_px - cx_img)
        y_px_new = cy_img + scale * (y_px - cy_img)

        # check if still inside image
        if x_px_new < 0 or x_px_new >= W or y_px_new < 0 or y_px_new >= H:
            # mark invisible if fully outside
            kpts_new[i, 2] = 0.0
            kpts_new[i, 0] = 0.0
            kpts_new[i, 1] = 0.0
        else:
            kpts_new[i, 0] = x_px_new / W
            kpts_new[i, 1] = y_px_new / H
            kpts_new[i, 2] = 1.0

    return scaled_img, cx_new, cy_new, w_new, h_new, kpts_new


# =========================
# Main driver
# =========================
def main():
    img_dir = DATA_ROOT / "images" / SPLIT
    lbl_dir = DATA_ROOT / "labels" / SPLIT
    vis_dir = DATA_ROOT / "vis" / SPLIT

    if not img_dir.is_dir() or not lbl_dir.is_dir():
        raise RuntimeError(f"Image or label dir not found: {img_dir}, {lbl_dir}")

    if MAKE_VIZ:
        vis_dir.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(p for p in img_dir.glob("*.png"))

    print(f"[info] Found {len(img_paths)} images in {img_dir}")

    for img_path in img_paths:
        stem = img_path.stem
        if SKIP_ALREADY_AUG and (ROLL_SUFFIX in stem or SCALE_SUFFIX in stem):
            # don't re-augment already augmented files
            continue

        lbl_path = lbl_dir / f"{stem}.txt"
        if not lbl_path.exists():
            print(f"[warn] No label for {img_path.name}, skipping.")
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[warn] Could not read image {img_path}, skipping.")
            continue

        try:
            cls_id, cx, cy, w, h, kpts = load_yolo_pose_label(lbl_path)
        except Exception as e:
            print(f"[warn] Bad label {lbl_path}: {e}")
            continue

        H, W = img.shape[:2]

        # ---------- ROLL AUGMENTATIONS ----------
        for shift_px in ROLL_SHIFTS:
            shift_px_mod = shift_px % W
            if shift_px_mod == 0:
                continue

            new_img, cx_new, cy_new, w_new, h_new, kpts_new = augment_roll(
                img, cls_id, cx, cy, w, h, kpts, shift_px_mod
            )

            # 1) Re-check bbox–keypoint constraint after roll
            if not all_kpts_ok_with_bbox(W, H, cx_new, cy_new, w_new, h_new, kpts_new):
                print(f"[skip-roll] {stem}, shift={shift_px_mod}: kpts too far from bbox")
                continue

            # 2) Check bbox fraction after roll
            # if w_new > BBOX_MAX_FRAC:
            #     print(f"[skip-roll] {stem}, shift={shift_px_mod}: bbox too wide ({w_new:.3f})")
            #     continue

            new_stem = f"{stem}_{ROLL_SUFFIX}{shift_px_mod}"
            out_img_path = img_dir / f"{new_stem}.png"
            out_lbl_path = lbl_dir / f"{new_stem}.txt"

            cv2.imwrite(str(out_img_path), new_img)
            save_yolo_pose_label(out_lbl_path, cls_id, cx_new, cy_new, w_new, h_new, kpts_new)

            if MAKE_VIZ:
                save_vis_image(vis_dir, new_stem, new_img, cx_new, cy_new, w_new, h_new, kpts_new)

        # ---------- SCALE AUGMENTATIONS ----------
        for s in SCALE_FACTORS:
            if abs(s - 1.0) < 1e-3:
                continue

            new_img, cx_new, cy_new, w_new, h_new, kpts_new = augment_scale(
                img, cls_id, cx, cy, w, h, kpts, s
            )

            # 1) Re-check bbox–keypoint constraint after scale
            if not all_kpts_ok_with_bbox(W, H, cx_new, cy_new, w_new, h_new, kpts_new):
                print(f"[skip-scale] {stem}, scale={s}: kpts too far from bbox")
                continue

            # 2) Check bbox fraction after scale
            if w_new > BBOX_MAX_FRAC:
                print(f"[skip-scale] {stem}, scale={s}: bbox too wide ({w_new:.3f})")
                continue

            s_tag = f"{s:.2f}".replace(".", "p")  # e.g. 0.85 -> "0p85"
            new_stem = f"{stem}_{SCALE_SUFFIX}{s_tag}"
            out_img_path = img_dir / f"{new_stem}.png"
            out_lbl_path = lbl_dir / f"{new_stem}.txt"

            cv2.imwrite(str(out_img_path), new_img)
            save_yolo_pose_label(
                out_lbl_path, cls_id, cx_new, cy_new, w_new, h_new, kpts_new
            )

            if MAKE_VIZ:
                save_vis_image(vis_dir, new_stem, new_img, cx_new, cy_new, w_new, h_new, kpts_new)

    print("[done] Augmentation finished.")


if __name__ == "__main__":
    main()
