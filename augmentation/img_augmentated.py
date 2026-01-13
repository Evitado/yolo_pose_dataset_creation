#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Offline augmenter for YOLO pose dataset (aircraft range images).

- Works on exported dataset structure:
    root/
      images/{train,val,test}/*.png
      labels/{train,val,test}/*.txt

- For each image/label pair:
    * creates several azimuth-roll (horizontal circular shift) variants
    * creates several scale (zoom in/out) variants
    * creates several small vertical-shift variants

Assumes label format (single line per file):
  class cx cy w h kp1x kp1y kp1v kp2x kp2y kp2v ...

All coords normalized to [0,1], visibility v in {0,1}.

We do NOT enforce:
  - keypoints near bbox
  - bbox width limit

Keypoints that leave the image during scaling/vertical shift are marked invisible (v=0).
"""

import os
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

# =========================
# CONFIG
# =========================
DATA_ROOT = Path("../aircraft_pose_all2")  # adjust if needed

# Now we can augment multiple splits
SPLITS = ["train","val"]  # often you want ["train"] only

# Horizontal circular shifts (in pixels) – azimuth rotations.
ROLL_SHIFTS = [
    16, 32, 64, 96, 128, 160, 192, 224,
    256, 320, 384, 448, 512, 576, 640,
    704, 768, 832, 896, 960, 1024, 1152,
    1280, 1408, 1536, 1664, 1792
]


# Uniform scale factors around image center:
# <1 = zoom out (aircraft smaller), >1 = zoom in (aircraft larger)
SCALE_FACTORS = [
     0.60, 0.75, 0.85,
    1.15, 1.25, 1.40, 1.60
]


# Small vertical translations in pixels (no wrap-around)
# negative = shift up, positive = shift down
VSHIFT_PIXELS = [ -8, 8]


# Name suffixes
ROLL_SUFFIX = "roll"
SCALE_SUFFIX = "scale"
VSHIFT_SUFFIX = "vshift"

# If True, skip images that already look augmented (contain suffix)
SKIP_ALREADY_AUG = False

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


def augment_vshift(
    img: np.ndarray,
    cls_id: int,
    cx: float,
    cy: float,
    w: float,
    h: float,
    kpts: np.ndarray,
    shift_px: int,
):
    """
    Vertical translation by shift_px pixels (no wraparound).

    - Image: cv2.warpAffine with translation (0, shift_px)
    - bbox cy: shifted by shift_px / H (clipped to [0,1])
    - keypoints y: shifted by shift_px / H
      keypoints that go out of [0,H) become invisible (v=0, x=y=0)
    """
    H, W = img.shape[:2]

    # ---- image: translate vertically, pad with zeros ----
    M = np.float32([[1, 0, 0],
                    [0, 1, float(shift_px)]])
    shifted_img = cv2.warpAffine(
        img, M, (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    # ---- bbox center (only cy changes) ----
    cy_px = cy * H
    cy_px_new = cy_px + shift_px
    cy_new = cy_px_new / H
    cy_new = float(np.clip(cy_new, 0.0, 1.0))

    # cx, w, h unchanged
    cx_new = cx
    w_new = w
    h_new = h

    # ---- keypoints ----
    kpts_new = kpts.copy()
    for i in range(kpts_new.shape[0]):
        x, y, v = kpts_new[i]
        if v <= 0.0:
            continue

        y_px = y * H
        y_px_new = y_px + shift_px

        # if outside image vertically → mark invisible
        if (y_px_new < 0) or (y_px_new >= H):
            kpts_new[i, 2] = 0.0
            kpts_new[i, 0] = 0.0
            kpts_new[i, 1] = 0.0
        else:
            kpts_new[i, 1] = y_px_new / H
            kpts_new[i, 2] = 1.0  # still visible

    return shifted_img, cx_new, cy_new, w_new, h_new, kpts_new


# =========================
# Per-split processing
# =========================
def process_split(split: str):
    img_dir = DATA_ROOT / "images" / split
    lbl_dir = DATA_ROOT / "labels" / split
    vis_dir = DATA_ROOT / "vis" / split

    if not img_dir.is_dir() or not lbl_dir.is_dir():
        print(f"[info] Split '{split}' not found (img: {img_dir}, lbl: {lbl_dir}), skipping.")
        return

    if MAKE_VIZ:
        vis_dir.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(p for p in img_dir.glob("*.png"))
    print(f"[info] Split '{split}': found {len(img_paths)} images in {img_dir}")

    for img_path in img_paths:
        stem = img_path.stem
        if SKIP_ALREADY_AUG and (ROLL_SUFFIX in stem or SCALE_SUFFIX in stem or VSHIFT_SUFFIX in stem):
            # don't re-augment already augmented files
            continue

        lbl_path = lbl_dir / f"{stem}.txt"
        if not lbl_path.exists():
            print(f"[warn] [{split}] No label for {img_path.name}, skipping.")
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[warn] [{split}] Could not read image {img_path}, skipping.")
            continue

        try:
            cls_id, cx, cy, w, h, kpts = load_yolo_pose_label(lbl_path)
        except Exception as e:
            print(f"[warn] [{split}] Bad label {lbl_path}: {e}")
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

        # ---------- VERTICAL SHIFT AUGMENTATIONS ----------
        for dy in VSHIFT_PIXELS:
            if dy == 0:
                continue

            new_img, cx_new, cy_new, w_new, h_new, kpts_new = augment_vshift(
                img, cls_id, cx, cy, w, h, kpts, dy
            )

            if dy >= 0:
                dy_tag = f"p{dy}"
            else:
                dy_tag = f"m{abs(dy)}"

            new_stem = f"{stem}_{VSHIFT_SUFFIX}{dy_tag}"
            out_img_path = img_dir / f"{new_stem}.png"
            out_lbl_path = lbl_dir / f"{new_stem}.txt"

            cv2.imwrite(str(out_img_path), new_img)
            save_yolo_pose_label(out_lbl_path, cls_id, cx_new, cy_new, w_new, h_new, kpts_new)

            if MAKE_VIZ:
                save_vis_image(vis_dir, new_stem, new_img, cx_new, cy_new, w_new, h_new, kpts_new)


# =========================
# Main driver
# =========================
def main():
    print(f"[info] DATA_ROOT = {DATA_ROOT}")
    for split in SPLITS:
        print(f"[info] === Processing split: {split} ===")
        process_split(split)
    print("[done] Augmentation finished for all requested splits.")


if __name__ == "__main__":
    main()
