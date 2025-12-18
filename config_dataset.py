#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Configuration for YOLO-pose dataset export from aircraft HDF5 files.
"""

from typing import Optional, Tuple

# =========================
# CONFIG — EDIT THESE
# =========================

# Data source: GCS prefix or local directory
SOURCE: str = (
    "gs://evitado_data/ml_datasets/thesis_femi_dataset_2025/verified_bags_v2"
)

# Output root directory for YOLO dataset
OUT_DIR: str = "./aircraft_pose_all"

# Dataset split (train, val, test) — must sum to 1.0
SPLIT: Tuple[float, float, float] = (0.8, 0.1, 0.1)

# Global RNG seed for reproducibility
RANDOM_SEED: int = 123

# Limit number of H5 files (for quick testing). Set to None to use all files.
MAX_H5_FILES: Optional[int] = None   # e.g. 4 or 5 for testing, or None for all

# =========================
# Image rendering
# =========================
DRAW_ON_OVERLAY: bool = False  # if True, tint aircraft mask red over RGB

# Image post-processing
APPLY_MEDIAN_FILTER: bool = True   # apply median filter on the final RGB image
MEDIAN_KSIZE: int = 3              # must be odd (3, 5, 7, ...)

# =========================
# Projection / transform (match your viewer)
# =========================
USE_TF_MATRIX: bool = True
APPLY_Z_FLIP: bool = False
KEYPOINT_AZ_SHIFT_COLS: int = 0
FLIP_VERTICAL_FOR_DRAW: bool = False
BASE_LINK_ROW_BAND: int = 1  # not used in projection, kept for potential future tuning

# =========================
# Keypoint rules
# =========================
SYN_KP_NAME: str = "front_wheels_mid"
REMOVE_KP_SET = {"center", "plane_front_left_wheel_link", "plane_front_right_wheel_link"}
FRONT_RIGHT_ALIASES = ["plane_front_right_wheel_link"]
FRONT_LEFT_ALIASES  = ["plane_front_left_wheel_link"]

# Visualization of keypoints
MAKE_VIZ: bool = True  # set False if you don't want vis/ outputs

# Keypoint vs bbox constraint (currently unused but kept for debugging)
KPT_BBOX_MARGIN_PX: int = 30  # allow keypoints to be this many pixels outside the bbox
ONLY_CHECK_VISIBLE_KPTS: bool = True  # only enforce constraint for v > 0 (2D-visible)

# Roll / azimuth shift when bbox is huge (mostly redundant if bbox>0.6 is skipped)
ROLL_WIDE_BBOX: bool = True
ROLL_WIDE_BBOX_FRAC: float = 0.9   # if bbox_w / W > this → roll
ROLL_WIDE_BBOX_COLS: int = 512     # columns to roll (circular, along width)

# ---- Ray-tracing style visibility check ----
RAY_VISIBILITY_CHECK: bool = True   # if False, fall back to simple in-bounds visibility
RAY_TOL: float = 1.5                # meters tolerance: kp is self-occluded if R_kp > R_hit + RAY_TOL
RAY_PATCH_RADIUS: int = 3           # 1 → 3×3, 2 → 5×5, 3 → 7×7, ...

# ---- 3D cluster adjustment for front_wheels_mid ----
# Start with raw midpoint between front-left and front-right wheel KPs, then:
#  - Check small radius around it (BASE_RADIUS) for enough aircraft points near ground.
#  - If too few, expand to EXPAND_RADIUS and use all points in that region as a cluster.
MID_BASE_RADIUS: float    = 1.0   # meters (small sphere around raw midpoint)
MID_EXPAND_RADIUS: float  = 3.0   # meters (larger sphere if first fails)
MID_Z_BAND: float         = 0.5   # consider points with z < z_min + MID_Z_BAND
MID_MIN_POINTS: int       = 6     # minimum #points to accept as a cluster
