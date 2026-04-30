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
    "/home/femi/Benchmarking_framework/Data/warning_b_test_h5"
)

# Output root directory for YOLO dataset
OUT_DIR: str = "./aircraft_pose_with_normalising_applied_multifield_only_3_4"

# Dataset split (train, val, test) — must sum to 1.0
SPLIT: Tuple[float, float, float] = (0.8, 0.1, 0.1)

# Global RNG seed for reproducibility
RANDOM_SEED: int = 123

# Limit number of H5 files (for quick testing). Set to None to use all files.
MAX_H5_FILES: Optional[int] = 1  # e.g. 4 or 5 for testing, or None for all

# Optional: reuse YOLO labels from an existing dataset/folder instead of
# regenerating them from projected keypoints.
# Supported inputs:
# - dataset root that contains labels/train|val|test
# - labels directory directly
# - any directory containing *.txt label files
REUSE_LABELS_FROM_DIR: str = ""
# If True, scenes are skipped when a matching external label is not found.
# If False, missing labels fall back to generated labels.
REUSE_LABELS_STRICT: bool = False

# =========================
# Image rendering
# =========================
DRAW_ON_OVERLAY: bool = False  # if True, tint aircraft mask red over RGB

# Output renderer for generated dataset images:
# - "single_field_colormap": use one lidar field + colormap
# - "multi_field_rgb": map three lidar fields directly into R/G/B
IMAGE_RENDER_MODE: str = "multi_field_rgb"

# Field mapping used only when IMAGE_RENDER_MODE="multi_field_rgb".
# Order is (R, G, B). Typical detailed choice:
#   ("reflectivity", "range", "intensity")
IMAGE_CHANNEL_FIELDS: Tuple[str, str, str] = ("reflectivity", "intensity", "range")
# Optional per-scene RGB balancing in multi-field mode.
# Disable this when one channel (e.g., range) is intentionally bright.
MULTI_FIELD_CHANNEL_BALANCE: bool = True
MULTI_FIELD_CHANNEL_BALANCE_MIN_GAIN: float = 0.25
MULTI_FIELD_CHANNEL_BALANCE_MAX_GAIN: float = 2.0

# Output colormap for single-field mode:
# - "grayscale" (dashcam-like)
# - "viridis"
# - "plasma"
# - "jet"
IMAGE_COLORMAP: str = "grayscale"

# Optional Retinex enhancement applied before colormap:
# - "off": no Retinex
# - "ssr": Single-Scale Retinex (uses first sigma)
# - "msr": Multi-Scale Retinex (averages across sigmas)
RETINEX_MODE: str = "off"
RETINEX_SIGMAS: Tuple[float, ...] = (15.0, 80.0, 250.0)
# Gain/offset on Retinex response before percentile normalization
RETINEX_GAIN: float = 1.0
RETINEX_OFFSET: float = 0.0
RETINEX_EPS: float = 1e-6
# Optional CLAHE after Retinex. By default this is applied only for SSR mode.
RETINEX_CLAHE_ENABLE: bool = True
RETINEX_CLAHE_ONLY_SSR: bool = True
RETINEX_CLAHE_CLIP_LIMIT: float = 2.0
RETINEX_CLAHE_TILE_GRID: int = 8

# Range-channel encoding in build_rgb_from_cols whenever a channel is mapped to "range":
# - "dashcam":           same dashcam normalization as reflectivity/intensity
# - "bright_band":       keep a distance band bright (configured below)
# - "inverse_log_range": near brighter, far darker
# - "log_range":         near darker, far brighter
# - "inverse_range":     stronger near-bright response
BLUE_CHANNEL_MODE: str = "bright_band"
# Parameters for "bright_band" mode:
# Distances in [RANGE_BRIGHT_MIN_M, RANGE_BRIGHT_MAX_M] are raised toward RANGE_BRIGHT_PEAK_LEVEL.
# Outside the band, intensity falls toward RANGE_BRIGHT_OUTSIDE_LEVEL with soft edges.
RANGE_BRIGHT_MIN_M: float = 2.0
RANGE_BRIGHT_MAX_M: float = 50.0
RANGE_BRIGHT_SOFT_EDGE_M: float = 8.0
RANGE_BRIGHT_OUTSIDE_LEVEL: float = 0.05
RANGE_BRIGHT_PEAK_LEVEL: float = 0.70

# Optional contrast shaping applied to the blue channel (power-law/gamma).
# 1.0 keeps values unchanged.
# <1.0 brightens low values; >1.0 darkens low values.
BLUE_CHANNEL_GAMMA: float = 1.2

# Optional: brighten non-range fields (e.g. intensity/reflectivity) where range
# mapping is bright, so far band emphasis is consistent across channels.
FAR_BRIGHT_BOOST_ENABLE: bool = True
FAR_BRIGHT_BOOST_FIELDS: Tuple[str, ...] = ("reflectivity", "intensity")
# 0.0 disables boost effect; 1.0 strongly pushes boosted pixels toward white.
FAR_BRIGHT_BOOST_STRENGTH: float = 0.45

# Optional global intensity boost (applied after normalization).
# 1.0 = no change, >1.0 = brighter intensity channel.
INTENSITY_BOOST_ENABLE: bool = True
INTENSITY_BOOST_GAIN: float = 1.35
# Optional row correction for near-sensor horizontal ring lines.
INTENSITY_ROW_CORRECTION_ENABLE: bool = True
# Apply the same row correction to reflectivity channel as well.
REFLECTIVITY_ROW_CORRECTION_ENABLE: bool = True
# 0.0 disables effect, 1.0 = full estimated row-bias removal.
INTENSITY_ROW_CORRECTION_STRENGTH: float = 0.80
# Smoothing of row-profile trend (in rows).
INTENSITY_ROW_CORRECTION_SIGMA_ROWS: float = 3.0
# Clamp per-row shift in normalized [0,1] units.
INTENSITY_ROW_CORRECTION_MAX_SHIFT: float = 0.12

# Export true single-channel PNGs (HxW) instead of 3-channel RGB.
# Keypoint labels/geometry are unchanged.
EXPORT_SINGLE_CHANNEL_IMAGE: bool = False
# Field used for single-channel export. If not present in an H5 scene,
# renderer falls back to first available among:
#   range -> signal -> intensity -> reflectivity -> ambient
SINGLE_CHANNEL_FIELD: str = "reflectivity"

# Optional: separate ground visually by darkening non-aircraft ground pixels.
# Uses `is_ground` mask from H5 when available.
GROUND_SEPARATION_ENABLE: bool = True
# 0.0 = black ground, 1.0 = no change
GROUND_ATTENUATION_FACTOR: float = 0.00
# Ground rendering mode:
# - "black": set detected ground pixels to zero
# - "attenuate": multiply by GROUND_ATTENUATION_FACTOR
GROUND_SEPARATION_MODE: str = "black"

# Optional RANSAC plane-based ground detection from xyz.
# This can fill gaps where `is_ground` mask misses many ground pixels.
GROUND_RANSAC_ENABLE: bool = True
# If True and `is_ground` exists, combine masks with OR; else use available mask.
GROUND_RANSAC_COMBINE_WITH_IS_GROUND: bool = True
GROUND_RANSAC_SAMPLE_MAX_POINTS: int = 50000
GROUND_RANSAC_ITERS: int = 120
GROUND_RANSAC_DIST_THRESH_M: float = 0.15
GROUND_RANSAC_MAX_TILT_DEG: float = 25.0
GROUND_RANSAC_MIN_INLIER_RATIO: float = 0.03
GROUND_RANSAC_MIN_POINTS: int = 3000

# Image post-processing
APPLY_MEDIAN_FILTER: bool = False  # dashcam-equivalent rendering uses no median filter
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
REMOVE_KP_SET = {"center", "plane_front_left_wheel_link", "plane_front_right_wheel_link","base_link","plane_rear_left_wheel_link",
    "plane_rear_right_wheel_link","left_wing_tip",
    "right_wing_tip"}
FRONT_RIGHT_ALIASES = ["plane_front_right_wheel_link"]
FRONT_LEFT_ALIASES  = ["plane_front_left_wheel_link"]

# Toggle: if True, use an explicit nose/front landing-gear keypoint when available
# for SYN_KP_NAME, instead of wheel-link midpoint fallback.
USE_NOSE_GEAR_CENTER_FOR_SYNTHETIC_FRONT_MID: bool = True
NOSE_GEAR_ALIASES = [
    "front_landing_gear",
    "front_gear",
    "nose_gear",
    "front_landing_gear_center",
    "nose_gear_center",
]

# Toggle: add two extra keypoints from warning-box style engine center names (if present)
ADD_ENGINE_WARNING_BOX_KEYPOINTS: bool = True
ENGINE_LEFT_KP_NAME: str = "engine_left_box_center"
ENGINE_RIGHT_KP_NAME: str = "engine_right_box_center"
ENGINE_LEFT_BOX_ALIASES = ["engine_left", "left_engine", "engine_l"]
ENGINE_RIGHT_BOX_ALIASES = ["engine_right", "right_engine", "engine_r"]


# Toggle: use warning-box YAML centers (resolved per aircraft profile) for
# front landing-gear and engine keypoints.
USE_WARNING_BOX_KEYPOINTS: bool = True
WARNING_PROFILE_CSV: str = "/home/femi/prof/outputs/warning_visualization_test_plan_profiles_only.csv"
WARNING_YAML_COLUMN: str = "recommended_yaml"
# Fallback YAML resolution (used when profile CSV has no row for a bag, e.g. 737_max8)
WARNING_YAML_ROOT: str = "/home/femi/evitado_description/aircraft_configs"
WARNING_YAML_RELPATH: str = "detection_configs/default.yaml"
WARNING_CENTER_KEYPOINT_NAME: str = "center"
WARNING_TARGET_LEVEL: int = 5
WARNING_CENTER_FRAME_OFFSET: Tuple[float, float, float] = (0.0, 0.0, 0.0)
WARNING_ENGINE_Z_OFFSET: float = 0.0
WARNING_LANDING_GEAR_Z_OFFSET: float = 0.0
WARNING_WING_Z_OFFSET: float = 0.0
WARNING_REAR_WING_Z_OFFSET: float = 0.0
WARNING_DERIVE_FRONT_GEAR_FROM_WHEELS: bool = True
WARNING_FRONT_GEAR_NAME_FILTERS = ["front_landing_gear", "landing_gear_front", "nose_gear", "front_gear"]
WARNING_ENGINE_LEFT_NAME_FILTERS = ["engine_left", "plane_engine_left", "left_engine"]
WARNING_ENGINE_RIGHT_NAME_FILTERS = ["engine_right", "plane_engine_right", "right_engine"]
# Engine keypoint refinement from warning boxes:
# 1) snap in 3D to aircraft points inside selected engine warning box.
ENGINE_BOX_SNAP_ENABLED: bool = True
ENGINE_BOX_SNAP_MIN_POINTS: int = 8
ENGINE_BOX_SNAP_EXPAND_FACTOR: float = 1.2
# If snapped 3D point is too far from the warning-box center, use box center instead.
# Set <= 0 to disable this guard.
ENGINE_BOX_SNAP_MAX_DRIFT_M: float = 1.25
# 2) snap in 2D to nearest valid aircraft pixel around projected location.
ENGINE_PIXEL_SNAP_ENABLED: bool = True
ENGINE_PIXEL_SNAP_RADIUS: int = 3
# If local pixel snap window has no aircraft pixels, optionally fall back to the
# nearest aircraft pixel in the full image (bounded by max distance).
ENGINE_PIXEL_SNAP_FALLBACK_TO_NEAREST: bool = False
ENGINE_PIXEL_SNAP_FALLBACK_MAX_DIST: int = 20
# Final pixel row bias for engine keypoints after snapping.
# Negative = move up, positive = move down.
ENGINE_PIXEL_ROW_BIAS: int = 0


# Visualization of keypoints
MAKE_VIZ: bool = True  # set False if you don't want vis/ outputs
# 3D debug export: save per-scene point cloud with keypoint markers as colored PLY.
DEBUG_POINTCLOUD_KEYPOINTS: bool = True
DEBUG_POINTCLOUD_MAX_SCENES: int = 20
DEBUG_POINTCLOUD_SAMPLE_EVERY_N: int = 1
DEBUG_POINTCLOUD_MAX_POINTS: int = 40000
# Optional live Open3D viewer (scene-by-scene) for the same debug cloud+keypoints.
# Close each window to continue to the next scene.
DEBUG_POINTCLOUD_LIVE_VIEWER: bool = True
# Draw visualization-only engine bounding boxes centered at EL/ER keypoints.
DRAW_ENGINE_VIS_BBOX: bool = True
ENGINE_VIS_BBOX_HALF_W: int = 24
ENGINE_VIS_BBOX_HALF_H: int = 16
# Draw visualization-only nose-gear bbox at best available nose keypoint.
DRAW_NOSE_GEAR_VIS_BBOX: bool = True
NOSE_VIS_BBOX_HALF_W: int = 20
NOSE_VIS_BBOX_HALF_H: int = 14

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
# For these keypoints, require at least one local aircraft depth hit in the
# ray patch; otherwise mark hidden. This helps suppress false-visible tips.
RAY_REQUIRE_LOCAL_HIT_KEYPOINTS = (
    # "left_wing_tip",
    # "right_wing_tip",
)
# Keypoints exempted from ray-based occlusion rejection.
# Useful for sparse/noisy masks where far points (wing tips / warning-box engines)
# can be falsely marked hidden by nearby clutter.
RAY_VISIBILITY_EXEMPT_KEYPOINTS = (
    # "plane_rear_left_wheel_link",
    # "plane_rear_right_wheel_link",
    "engine_left_box_center",
    "engine_right_box_center",
)

# ---- 3D cluster adjustment for front_wheels_mid ----
# Start with raw midpoint between front-left and front-right wheel KPs, then:
#  - Check small radius around it (BASE_RADIUS) for enough aircraft points near ground.
#  - If too few, expand to EXPAND_RADIUS and use all points in that region as a cluster.
MID_BASE_RADIUS: float    = 1.0   # meters (small sphere around raw midpoint)
MID_EXPAND_RADIUS: float  = 3.0   # meters (larger sphere if first fails)
MID_Z_BAND: float         = 0.5   # consider points with z < z_min + MID_Z_BAND
MID_MIN_POINTS: int       = 6     # minimum #points to accept as a cluster
