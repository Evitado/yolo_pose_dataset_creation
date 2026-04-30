#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Offline augmenter for YOLO pose dataset (aircraft range images).

Reads from DATA_ROOT and writes a NEW augmented dataset to OUT_ROOT.

Input structure:
  DATA_ROOT/
    images/{train,val,test}/*.png
    labels/{train,val,test}/*.txt

Output structure:
  OUT_ROOT/
    images/{train,val,test}/*.png
    labels/{train,val,test}/*.txt
    vis/{train,val,test}/*.png   (optional)

Augmentations implemented:
  - azimuth-roll (horizontal circular shift): np.roll along axis=1
  - scale (zoom in/out): warpAffine about image center
  - vertical shift: warpAffine translate in y

Label format (single line per file):
  class cx cy w h kp1x kp1y kp1v kp2x kp2y kp2v ...

All coords normalized to [0,1], visibility v in {0,1}.
Keypoints that leave the image are marked invisible (v=0, x=y=0).

Recommended:
  - Augment TRAIN only (do NOT augment val/test for fair evaluation).
"""

from pathlib import Path
from typing import Tuple
import argparse

import cv2
import numpy as np

# =========================
# CONFIG
# =========================
DATA_ROOT = Path("/home/femi/yolo_pose_dataset_creation/aircraft_pose_with_normalising_applied_multifield_only_3_2")        # INPUT dataset
OUT_ROOT  = Path("/home/femi/yolo_pose_dataset_creation/aircraft_pose_with_normalising_applied_multifield_only_3_2aug_2")    # OUTPUT dataset (NEW)

# Usually augment TRAIN only
SPLITS = ["train"]

# Horizontal circular shifts (pixels). For W=1024, these are reasonable.
ROLL_SHIFTS = [2,4,8,16,32,64,128, 256, 384, 512, 640, 768]

# Uniform scale factors around image center:
# <1 = zoom out (aircraft smaller), >1 = zoom in (aircraft larger)
SCALE_FACTORS = [0.9,1.1]  # e.g. [0.9, 1.1]

# Vertical translations in pixels (no wrap-around)
VSHIFT_PIXELS = [2,-2]  # e.g. [-8, 8]

# Name suffixes
ROLL_SUFFIX = "roll"
SCALE_SUFFIX = "scale"
VSHIFT_SUFFIX = "vshift"

# If True, skip inputs that already look augmented (contain suffix)
SKIP_ALREADY_AUG = False

# If True, create visualization images (bbox + kpts) under OUT_ROOT/vis/<split>/
MAKE_VIZ = True

# If True, copy original (non-augmented) image+label into OUT_ROOT as well
# so OUT_ROOT is a complete dataset on its own.
COPY_ORIGINALS = True


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
    w  = float(parts[3]); h  = float(parts[4])
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
) -> None:
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
        parts.append(str(int(round(float(v)))))
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
) -> None:
    """
    Save a visualization image with bbox and keypoints drawn.
    """
    H, W = img.shape[:2]

    # bbox in px
    x1 = int((cx - w / 2.0) * W)
    y1 = int((cy - h / 2.0) * H)
    x2 = int((cx + w / 2.0) * W)
    y2 = int((cy + h / 2.0) * H)

    # clamp
    x1 = max(0, min(W - 1, x1))
    x2 = max(0, min(W - 1, x2))
    y1 = max(0, min(H - 1, y1))
    y2 = max(0, min(H - 1, y2))

    vis_img = img.copy()

    # bbox (green)
    cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # kpts (red)
    for idx, (x, y, v) in enumerate(kpts):
        if float(v) <= 0.0:
            continue
        x_px = int(float(x) * W)
        y_px = int(float(y) * H)
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
    cx: float,
    cy: float,
    w: float,
    h: float,
    kpts: np.ndarray,
    shift_px: int
):
    """
    Horizontal circular roll by shift_px columns.
    """
    H, W = img.shape[:2]
    rolled_img = np.roll(img, shift=shift_px, axis=1)

    # bbox center x shift
    cx_px = cx * W
    cx_px_new = (cx_px + shift_px) % W
    cx_new = cx_px_new / W

    # keypoints x shift
    kpts_new = kpts.copy()
    for i in range(kpts_new.shape[0]):
        x, y, v = kpts_new[i]
        if float(v) <= 0.0:
            continue
        x_px = float(x) * W
        x_px_new = (x_px + shift_px) % W
        kpts_new[i, 0] = x_px_new / W

    return rolled_img, float(cx_new), float(cy), float(w), float(h), kpts_new


def augment_scale(
    img: np.ndarray,
    cx: float,
    cy: float,
    w: float,
    h: float,
    kpts: np.ndarray,
    scale: float
):
    """
    Uniform scale around image center.
    """
    H, W = img.shape[:2]
    cx_img = W / 2.0
    cy_img = H / 2.0

    M = cv2.getRotationMatrix2D((cx_img, cy_img), 0.0, scale)
    scaled_img = cv2.warpAffine(
        img, M, (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    # bbox
    cx_px = cx * W
    cy_px = cy * H
    w_px  = w * W
    h_px  = h * H

    cx_px_new = cx_img + scale * (cx_px - cx_img)
    cy_px_new = cy_img + scale * (cy_px - cy_img)
    w_px_new  = w_px * scale
    h_px_new  = h_px * scale

    cx_new = float(np.clip(cx_px_new / W, 0.0, 1.0))
    cy_new = float(np.clip(cy_px_new / H, 0.0, 1.0))
    w_new  = float(np.clip(w_px_new / W, 1e-6, 1.0))
    h_new  = float(np.clip(h_px_new / H, 1e-6, 1.0))

    # keypoints
    kpts_new = kpts.copy()
    for i in range(kpts_new.shape[0]):
        x, y, v = kpts_new[i]
        if float(v) <= 0.0:
            continue

        x_px = float(x) * W
        y_px = float(y) * H

        x_px_new = cx_img + scale * (x_px - cx_img)
        y_px_new = cy_img + scale * (y_px - cy_img)

        if x_px_new < 0 or x_px_new >= W or y_px_new < 0 or y_px_new >= H:
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
    cx: float,
    cy: float,
    w: float,
    h: float,
    kpts: np.ndarray,
    shift_px: int,
):
    """
    Vertical translation by shift_px pixels (no wraparound).
    """
    H, W = img.shape[:2]
    M = np.float32([[1, 0, 0],
                    [0, 1, float(shift_px)]])
    shifted_img = cv2.warpAffine(
        img, M, (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    cy_px = cy * H
    cy_px_new = cy_px + shift_px
    cy_new = float(np.clip(cy_px_new / H, 0.0, 1.0))

    kpts_new = kpts.copy()
    for i in range(kpts_new.shape[0]):
        x, y, v = kpts_new[i]
        if float(v) <= 0.0:
            continue

        y_px = float(y) * H
        y_px_new = y_px + shift_px

        if (y_px_new < 0) or (y_px_new >= H):
            kpts_new[i, 2] = 0.0
            kpts_new[i, 0] = 0.0
            kpts_new[i, 1] = 0.0
        else:
            kpts_new[i, 1] = y_px_new / H
            kpts_new[i, 2] = 1.0

    return shifted_img, float(cx), cy_new, float(w), float(h), kpts_new


# =========================
# Per-split processing
# =========================
def process_split(split: str) -> None:
    in_img_dir = DATA_ROOT / "images" / split
    in_lbl_dir = DATA_ROOT / "labels" / split

    if not in_img_dir.is_dir() or not in_lbl_dir.is_dir():
        print(f"[info] Split '{split}' not found (img: {in_img_dir}, lbl: {in_lbl_dir}), skipping.")
        return

    out_img_dir = OUT_ROOT / "images" / split
    out_lbl_dir = OUT_ROOT / "labels" / split
    out_vis_dir = OUT_ROOT / "vis" / split

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)
    if MAKE_VIZ:
        out_vis_dir.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(in_img_dir.glob("*.png"))
    print(f"[info] Split '{split}': found {len(img_paths)} images in {in_img_dir}")
    print(f"[info] Writing output to: {OUT_ROOT.resolve()}")

    for img_path in img_paths:
        stem = img_path.stem

        if SKIP_ALREADY_AUG and (ROLL_SUFFIX in stem or SCALE_SUFFIX in stem or VSHIFT_SUFFIX in stem):
            continue

        lbl_path = in_lbl_dir / f"{stem}.txt"
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

        # ---- copy original into OUT_ROOT (optional) ----
        if COPY_ORIGINALS:
            out_img0 = out_img_dir / f"{stem}.png"
            out_lbl0 = out_lbl_dir / f"{stem}.txt"
            if not out_img0.exists():
                cv2.imwrite(str(out_img0), img)
            if not out_lbl0.exists():
                save_yolo_pose_label(out_lbl0, cls_id, cx, cy, w, h, kpts)
            if MAKE_VIZ:
                save_vis_image(out_vis_dir, stem, img, cx, cy, w, h, kpts)

        H, W = img.shape[:2]

        # ---------- ROLL ----------
        for shift_px in ROLL_SHIFTS:
            shift_px_mod = int(shift_px) % W
            if shift_px_mod == 0:
                continue

            new_img, cx_new, cy_new, w_new, h_new, kpts_new = augment_roll(
                img, cx, cy, w, h, kpts, shift_px_mod
            )

            new_stem = f"{stem}_{ROLL_SUFFIX}{shift_px_mod}"
            out_img_path = out_img_dir / f"{new_stem}.png"
            out_lbl_path = out_lbl_dir / f"{new_stem}.txt"

            cv2.imwrite(str(out_img_path), new_img)
            save_yolo_pose_label(out_lbl_path, cls_id, cx_new, cy_new, w_new, h_new, kpts_new)

            if MAKE_VIZ:
                save_vis_image(out_vis_dir, new_stem, new_img, cx_new, cy_new, w_new, h_new, kpts_new)

        # ---------- SCALE ----------
        for s in SCALE_FACTORS:
            if abs(float(s) - 1.0) < 1e-3:
                continue

            new_img, cx_new, cy_new, w_new, h_new, kpts_new = augment_scale(
                img, cx, cy, w, h, kpts, float(s)
            )

            s_tag = f"{float(s):.2f}".replace(".", "p")
            new_stem = f"{stem}_{SCALE_SUFFIX}{s_tag}"
            out_img_path = out_img_dir / f"{new_stem}.png"
            out_lbl_path = out_lbl_dir / f"{new_stem}.txt"

            cv2.imwrite(str(out_img_path), new_img)
            save_yolo_pose_label(out_lbl_path, cls_id, cx_new, cy_new, w_new, h_new, kpts_new)

            if MAKE_VIZ:
                save_vis_image(out_vis_dir, new_stem, new_img, cx_new, cy_new, w_new, h_new, kpts_new)

        # ---------- VSHIFT ----------
        for dy in VSHIFT_PIXELS:
            dy = int(dy)
            if dy == 0:
                continue

            new_img, cx_new, cy_new, w_new, h_new, kpts_new = augment_vshift(
                img, cx, cy, w, h, kpts, dy
            )

            dy_tag = f"p{dy}" if dy >= 0 else f"m{abs(dy)}"
            new_stem = f"{stem}_{VSHIFT_SUFFIX}{dy_tag}"
            out_img_path = out_img_dir / f"{new_stem}.png"
            out_lbl_path = out_lbl_dir / f"{new_stem}.txt"

            cv2.imwrite(str(out_img_path), new_img)
            save_yolo_pose_label(out_lbl_path, cls_id, cx_new, cy_new, w_new, h_new, kpts_new)

            if MAKE_VIZ:
                save_vis_image(out_vis_dir, new_stem, new_img, cx_new, cy_new, w_new, h_new, kpts_new)


# =========================
# Main driver
# =========================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline augmenter for YOLO pose dataset."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT,
        help=f"Input dataset root (default: {DATA_ROOT})",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=OUT_ROOT,
        help=f"Output dataset root (default: {OUT_ROOT})",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val", "test"],
        default=SPLITS,
        help=f"Dataset splits to process (default: {' '.join(SPLITS)})",
    )
    parser.add_argument(
        "--allow-same-root",
        action="store_true",
        help="Allow writing to the same root as --data-root (disabled by default).",
    )
    return parser.parse_args()


def main() -> None:
    global DATA_ROOT, OUT_ROOT

    args = parse_args()
    DATA_ROOT = args.data_root.expanduser().resolve()
    OUT_ROOT = args.out_root.expanduser().resolve()

    if DATA_ROOT == OUT_ROOT and not args.allow_same_root:
        raise SystemExit(
            "[error] DATA_ROOT and OUT_ROOT are the same. "
            "Set --out-root to another folder or use --allow-same-root."
        )

    print(f"[info] DATA_ROOT = {DATA_ROOT}")
    print(f"[info] OUT_ROOT  = {OUT_ROOT}")
    for split in args.splits:
        print(f"[info] === Processing split: {split} ===")
        process_split(split)
    print("[done] Augmentation finished for all requested splits.")


if __name__ == "__main__":
    main()
