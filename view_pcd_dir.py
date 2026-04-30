#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
View point-cloud files from a directory sequentially.

Examples:
  python view_pcd_dir.py
  python view_pcd_dir.py --root ./pcd_from_yolo --pattern a350 --max 20
  python view_pcd_dir.py --root ./pcd_from_yolo --recursive --shuffle
  python view_pcd_dir.py --file ./pcd_from_yolo/scene_001.pcd
"""

from __future__ import annotations

import argparse
import ast
import csv
import math
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Runtime toggles (edit in code instead of CLI flags)
WARNING_PASS_FAIL_ENABLED: bool = True
# If detected engine-left and engine-right regions overlap heavily, collapse to one region.
ENGINE_LR_OVERLAP_AS_ONE_RATIO_THR: float = 0.60

try:
    from config_dataset import (
        SOURCE,
        WARNING_PROFILE_CSV,
        WARNING_YAML_COLUMN,
        WARNING_YAML_ROOT,
        WARNING_YAML_RELPATH,
        WARNING_TARGET_LEVEL,
        WARNING_CENTER_FRAME_OFFSET,
        WARNING_ENGINE_Z_OFFSET,
        WARNING_LANDING_GEAR_Z_OFFSET,
        WARNING_WING_Z_OFFSET,
        WARNING_REAR_WING_Z_OFFSET,
        WARNING_FRONT_GEAR_NAME_FILTERS,
        WARNING_ENGINE_LEFT_NAME_FILTERS,
        WARNING_ENGINE_RIGHT_NAME_FILTERS,
    )
except Exception:
    SOURCE = ""
    WARNING_PROFILE_CSV = ""
    WARNING_YAML_COLUMN = "recommended_yaml"
    WARNING_YAML_ROOT = ""
    WARNING_YAML_RELPATH = "detection_configs/default.yaml"
    WARNING_TARGET_LEVEL = 5
    WARNING_CENTER_FRAME_OFFSET = (0.0, 0.0, 0.0)
    WARNING_ENGINE_Z_OFFSET = 0.0
    WARNING_LANDING_GEAR_Z_OFFSET = 0.0
    WARNING_WING_Z_OFFSET = 0.0
    WARNING_REAR_WING_Z_OFFSET = 0.0
    WARNING_FRONT_GEAR_NAME_FILTERS = ["front_landing_gear", "landing_gear_front", "nose_gear", "front_gear"]
    WARNING_ENGINE_LEFT_NAME_FILTERS = ["engine_left", "plane_engine_left", "left_engine"]
    WARNING_ENGINE_RIGHT_NAME_FILTERS = ["engine_right", "plane_engine_right", "right_engine"]


def _collect_cloud_files(
    root: Path,
    recursive: bool,
    pattern: str,
    exts: List[str],
) -> List[Path]:
    if not root.exists() or not root.is_dir():
        raise RuntimeError(f"Directory not found: {root}")

    norm_exts = {("." + e.lower().lstrip(".")) for e in exts if e.strip()}
    iterator = root.rglob("*") if recursive else root.glob("*")
    files = [p for p in iterator if p.is_file() and p.suffix.lower() in norm_exts]

    if pattern:
        p = pattern.lower()
        files = [f for f in files if p in f.name.lower()]

    return sorted(files)


def _print_cloud_summary(path: Path, xyz: np.ndarray) -> None:
    if xyz.size == 0:
        print(f"[empty] {path}")
        return
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    center = (mins + maxs) * 0.5
    print(
        f"  points={xyz.shape[0]} "
        f"min=({mins[0]:.3f},{mins[1]:.3f},{mins[2]:.3f}) "
        f"max=({maxs[0]:.3f},{maxs[1]:.3f},{maxs[2]:.3f}) "
        f"center=({center[0]:.3f},{center[1]:.3f},{center[2]:.3f})"
    )


def _split_main_and_keypoints(xyz: np.ndarray, kpt_count: int) -> Tuple[np.ndarray, np.ndarray]:
    if xyz.size == 0:
        return xyz, xyz
    n = int(xyz.shape[0])
    k = int(max(0, kpt_count))
    if k <= 0 or n <= k:
        return xyz, xyz[:0]
    return xyz[: n - k], xyz[n - k :]


def _read_xyz_from_ascii_pcd(path: Path) -> np.ndarray:
    points_n: Optional[int] = None
    header_lines = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            header_lines += 1
            s = line.strip().lower()
            if s.startswith("points"):
                parts = s.split()
                if len(parts) >= 2:
                    try:
                        points_n = int(parts[1])
                    except Exception:
                        points_n = None
            if s.startswith("data"):
                if "ascii" not in s:
                    raise RuntimeError("Only ASCII .pcd is supported in --no-vis mode without open3d.")
                break
        else:
            raise RuntimeError("Invalid .pcd header (missing DATA line).")

    arr = np.loadtxt(str(path), dtype=np.float64, skiprows=header_lines, max_rows=points_n)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 3:
        raise RuntimeError(".pcd does not contain xyz columns.")
    return np.asarray(arr[:, :3], dtype=np.float64)


def _read_xyz_from_ascii_ply(path: Path) -> np.ndarray:
    n_vertices: Optional[int] = None
    header_lines = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        first = f.readline()
        header_lines += 1
        if "ply" not in first.lower():
            raise RuntimeError("Not a PLY file.")
        for line in f:
            header_lines += 1
            s = line.strip().lower()
            if s.startswith("format") and "ascii" not in s:
                raise RuntimeError("Only ASCII .ply is supported in --no-vis mode without open3d.")
            if s.startswith("element vertex"):
                parts = s.split()
                if len(parts) >= 3:
                    try:
                        n_vertices = int(parts[2])
                    except Exception:
                        n_vertices = None
            if s == "end_header":
                break
        else:
            raise RuntimeError("Invalid .ply header (missing end_header).")

    arr = np.loadtxt(str(path), dtype=np.float64, skiprows=header_lines, max_rows=n_vertices)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 3:
        raise RuntimeError(".ply does not contain xyz columns.")
    return np.asarray(arr[:, :3], dtype=np.float64)


def _read_xyz(path: Path, o3d: Any) -> np.ndarray:
    if o3d is not None:
        pcd = o3d.io.read_point_cloud(str(path))
        if pcd.is_empty():
            return np.empty((0, 3), dtype=np.float64)
        xyz = np.asarray(pcd.points, dtype=np.float64)
        return xyz if xyz.ndim == 2 and xyz.shape[1] >= 3 else np.empty((0, 3), dtype=np.float64)

    suf = path.suffix.lower()
    if suf == ".pcd":
        return _read_xyz_from_ascii_pcd(path)
    if suf == ".ply":
        return _read_xyz_from_ascii_ply(path)
    raise RuntimeError(f"Unsupported extension for fallback reader: {path.suffix}")


def _warning_color_rgb01(name: str) -> np.ndarray:
    n = str(name).lower()
    # Fixed role colors for easier visual reading across aircraft:
    # - engines: pink
    # - nose/front gear: blue
    if "engine" in n:
        return np.array([1.0, 0.30, 0.80], dtype=np.float64)
    if "landing_gear" in n or "front" in n or "nose" in n:
        return np.array([0.20, 0.50, 1.00], dtype=np.float64)
    return np.array([1.0, 0.35, 0.1], dtype=np.float64)


def _warning_edges() -> List[Tuple[int, int]]:
    return [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]


def _oriented_box_corners(center_world: np.ndarray, half: np.ndarray, R_world_to_box: np.ndarray) -> np.ndarray:
    hx, hy, hz = [float(max(v, 1e-9)) for v in np.asarray(half, dtype=np.float64).reshape(3)]
    offs_box = np.array(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ],
        dtype=np.float64,
    )
    c = np.asarray(center_world, dtype=np.float64).reshape(1, 3)
    Rwb = np.asarray(R_world_to_box, dtype=np.float64).reshape(3, 3)
    return c + (offs_box @ Rwb.T)


def _point_inside_warning_box(p_world: np.ndarray, spec: Dict[str, Any], tol: float = 1e-6) -> Tuple[bool, np.ndarray]:
    c = np.asarray(spec["center_world"], dtype=np.float64).reshape(1, 3)
    Rwb = np.asarray(spec["R_world_to_box"], dtype=np.float64).reshape(3, 3)
    h = np.asarray(spec["half"], dtype=np.float64).reshape(3)
    local = (np.asarray(p_world, dtype=np.float64).reshape(1, 3) - c) @ Rwb
    inside = bool(np.all(np.abs(local[0]) <= (h + float(tol))))
    return inside, local.reshape(3)


def _inside_mask_warning_box(points_world: np.ndarray, spec: Dict[str, Any], tol: float = 1e-6) -> np.ndarray:
    pts = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    if pts.size == 0:
        return np.zeros((0,), dtype=bool)
    c = np.asarray(spec["center_world"], dtype=np.float64).reshape(1, 3)
    Rwb = np.asarray(spec["R_world_to_box"], dtype=np.float64).reshape(3, 3)
    h = np.asarray(spec["half"], dtype=np.float64).reshape(1, 3)
    local = (pts - c) @ Rwb
    return np.all(np.abs(local) <= (h + float(tol)), axis=1)


def _unique_quantized_point_keys(points_world: np.ndarray, quant_scale: float = 1000.0) -> np.ndarray:
    pts = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    if pts.size == 0:
        return np.empty(
            (0,),
            dtype=np.dtype([("x", np.int64), ("y", np.int64), ("z", np.int64)]),
        )
    q = np.rint(pts * float(max(1e-9, quant_scale))).astype(np.int64)
    keys = np.ascontiguousarray(q).view(
        np.dtype([("x", np.int64), ("y", np.int64), ("z", np.int64)])
    ).reshape(-1)
    return np.unique(keys)


def _count_quantized_point_overlap(
    points_a: np.ndarray,
    points_b: np.ndarray,
    quant_scale: float = 1000.0,
) -> Tuple[int, int, int]:
    keys_a = _unique_quantized_point_keys(points_a, quant_scale=quant_scale)
    keys_b = _unique_quantized_point_keys(points_b, quant_scale=quant_scale)
    if keys_a.size == 0 or keys_b.size == 0:
        return 0, int(keys_a.size), int(keys_b.size)
    inter = np.intersect1d(keys_a, keys_b, assume_unique=True)
    return int(inter.size), int(keys_a.size), int(keys_b.size)


def _norm_key(v: str | None) -> str:
    s = str(v or "").strip().lower()
    s = re.sub(r"[\s\-]+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _extract_aircraft_token_from_bag(bag_name: str) -> str:
    m = re.match(r"^movement_(.+?)(?:__|$)", str(bag_name), flags=re.IGNORECASE)
    if not m:
        return str(bag_name)
    return m.group(1).strip("_")


def _slugify_aircraft_name(name: str) -> str:
    s = str(name).strip().lower()
    s = re.sub(r"[\s\-]+", "_", s)
    s = re.sub(r"[^a-z0-9_+]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _iter_warning_yaml_slug_candidates(bag_name: str) -> List[str]:
    seed = _extract_aircraft_token_from_bag(bag_name)
    out: List[str] = []
    seen: set[str] = set()

    def _add(v: str | None) -> None:
        if not v:
            return
        s = _slugify_aircraft_name(v)
        if not s or s in seen:
            return
        seen.add(s)
        out.append(s)

    base = str(seed).strip()
    if base:
        _add(base)
        _add(base.replace("_", " "))
        _add(base.replace("_", "-"))
        _add(re.sub(r"([A-Za-z])(\d)", r"\1_\2", base))
        _add(re.sub(r"(\d)([A-Za-z])", r"\1_\2", base))
        if re.match(r"^[bB]\d", base):
            b = base[1:]
            _add(b)
            _add(b.replace("_", " "))
            _add(b.replace("_", "-"))

    expanded: List[str] = []
    seen2: set[str] = set()
    for s in out:
        variants = [s, s.replace("max_", "max")]
        if s.endswith("_split"):
            variants.append(s[: -len("_split")])
        for v in variants:
            if v and v not in seen2:
                seen2.add(v)
                expanded.append(v)
    return expanded


def _load_warning_profile_map(csv_path: str) -> Dict[str, Dict[str, str]]:
    p = Path(csv_path).expanduser()
    if not p.exists() or not p.is_file():
        return {}
    out: Dict[str, Dict[str, str]] = {}
    with p.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not isinstance(row, dict):
                continue
            keys = {
                _norm_key(row.get("profile_name", "")),
                _norm_key(row.get("aircraft", "")),
            }
            for k in keys:
                if k and k not in out:
                    out[k] = row
    return out


def _lookup_warning_profile_entry(mapping: Dict[str, Dict[str, str]], bag_name: str) -> Optional[Dict[str, str]]:
    if not mapping:
        return None
    token = _norm_key(_extract_aircraft_token_from_bag(bag_name))
    cands: List[str] = []
    if token:
        cands.append(token)
        if token.endswith("_split"):
            cands.append(token[: -len("_split")])
        cands.append(token.replace("_split", ""))
        cands.append(token.replace("split_", ""))
        cands.append(token.replace("_split_", "_"))
    # Keep order while de-duplicating.
    seen: set[str] = set()
    for k in cands:
        if not k or k in seen:
            continue
        seen.add(k)
        if k in mapping:
            return mapping[k]
    return None


def _parse_csv_box_name_filters(row: Optional[Dict[str, str]]) -> List[str]:
    if not row:
        return []
    raw_py = str(row.get("box_name_filters_py", "") or "").strip()
    if raw_py:
        try:
            obj = ast.literal_eval(raw_py)
            if isinstance(obj, list):
                return [str(x).strip() for x in obj if str(x).strip()]
        except Exception:
            pass
    raw = str(row.get("box_name_filters", "") or "").strip()
    if raw:
        return [x.strip() for x in raw.split("|") if x.strip()]
    return []


def _resolve_warning_yaml_for_bag(
    bag_name: str,
    warning_yaml_root: str,
    warning_yaml_relpath: str,
) -> Optional[Path]:
    root = Path(warning_yaml_root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        return None
    rel = str(warning_yaml_relpath or "detection_configs/default.yaml").strip("/")
    for slug in _iter_warning_yaml_slug_candidates(bag_name=bag_name):
        p = root / slug / rel
        if p.exists():
            return p
    return None


def _load_yaml_crop_boxes(yaml_path: Path, cache: Dict[Path, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    yp = yaml_path.expanduser().resolve()
    if yp in cache:
        return cache[yp]
    try:
        import yaml
    except Exception:
        cache[yp] = []
        return cache[yp]

    if not yp.exists():
        cache[yp] = []
        return cache[yp]

    try:
        data = yaml.safe_load(yp.read_text(encoding="utf-8"))
        boxes: List[Dict[str, Any]] = []
        if isinstance(data, dict) and isinstance(data.get("crop_boxes"), dict):
            for name, box in data["crop_boxes"].items():
                if not isinstance(box, dict):
                    continue
                try:
                    boxes.append(
                        {
                            "name": str(name),
                            "warning_level": int(box.get("warning_level", -1)),
                            "x": float(box["x"]),
                            "y": float(box["y"]),
                            "z": float(box["z"]),
                            "rx": float(box.get("rx", 0.0)),
                            "ry": float(box.get("ry", 0.0)),
                            "rz": float(box.get("rz", 0.0)),
                            "sx": float(box.get("sx", 0.0)),
                            "sy": float(box.get("sy", 0.0)),
                            "sz": float(box.get("sz", 0.0)),
                        }
                    )
                except Exception:
                    continue
        cache[yp] = boxes
        return boxes
    except Exception:
        cache[yp] = []
        return cache[yp]


def _euler_xyz_to_rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    rx_mat = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cx, -sx],
            [0.0, sx, cx],
        ],
        dtype=np.float64,
    )
    ry_mat = np.array(
        [
            [cy, 0.0, sy],
            [0.0, 1.0, 0.0],
            [-sy, 0.0, cy],
        ],
        dtype=np.float64,
    )
    rz_mat = np.array(
        [
            [cz, -sz, 0.0],
            [sz, cz, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return rz_mat @ ry_mat @ rx_mat


def _get_warning_local_z_adjustment(box_name: str) -> float:
    n = str(box_name).lower()
    if "engine" in n:
        return float(WARNING_ENGINE_Z_OFFSET)
    if "landing_gear" in n or "front_landing_gear" in n or "gear" in n:
        return float(WARNING_LANDING_GEAR_Z_OFFSET)
    if "rear_wing" in n:
        return float(WARNING_REAR_WING_Z_OFFSET)
    if "wing" in n:
        return float(WARNING_WING_Z_OFFSET)
    return 0.0


def _box_center_local(b: Dict[str, Any]) -> np.ndarray:
    p = np.array([float(b["x"]), float(b["y"]), float(b["z"])], dtype=np.float64)
    p[2] += _get_warning_local_z_adjustment(str(b.get("name", "")))
    return p


def _pick_box_by_filters(boxes: List[Dict[str, Any]], filters: List[str]) -> Optional[Dict[str, Any]]:
    fl = [str(x).lower() for x in filters]
    for b in boxes:
        n = str(b.get("name", "")).lower()
        if any(f in n for f in fl):
            return b
    return None


def _pick_engine_side_by_name(boxes: List[Dict[str, Any]], side: str) -> Optional[Dict[str, Any]]:
    side_s = str(side).strip().lower()
    if side_s not in ("left", "right"):
        return None
    cands: List[Dict[str, Any]] = []
    for b in boxes:
        n = str(b.get("name", "")).lower()
        if "engine" not in n:
            continue
        if side_s in n and ("left" in n or "right" in n):
            cands.append(b)
    if not cands:
        return None
    return max(cands, key=lambda bb: abs(float(bb.get("y", 0.0))))


def _pick_engine_lr_boxes(boxes: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    eng_like: List[Dict[str, Any]] = []
    for b in boxes:
        n = str(b.get("name", "")).lower()
        if "engine" in n:
            eng_like.append(b)

    # Prefer explicit side names (engine_left/engine1_left/etc.) before y-sign.
    left_box = _pick_engine_side_by_name(eng_like, "left")
    right_box = _pick_engine_side_by_name(eng_like, "right")

    if eng_like:
        left_candidates = [
            b for b in eng_like
            if float(b.get("y", 0.0)) >= 0.0
            and (right_box is None or str(b.get("name", "")) != str(right_box.get("name", "")))
        ]
        right_candidates = [
            b for b in eng_like
            if float(b.get("y", 0.0)) < 0.0
            and (left_box is None or str(b.get("name", "")) != str(left_box.get("name", "")))
        ]
        if left_box is None and left_candidates:
            left_box = max(left_candidates, key=lambda bb: abs(float(bb.get("y", 0.0))))
        if right_box is None and right_candidates:
            right_box = max(right_candidates, key=lambda bb: abs(float(bb.get("y", 0.0))))

    if left_box is None:
        left_box = _pick_box_by_filters(boxes, list(WARNING_ENGINE_LEFT_NAME_FILTERS))
    if right_box is None:
        right_box = _pick_box_by_filters(boxes, list(WARNING_ENGINE_RIGHT_NAME_FILTERS))

    if left_box is not None and right_box is not None:
        same_name = str(left_box.get("name", "")) == str(right_box.get("name", ""))
        same_xyz = (
            abs(float(left_box.get("x", 0.0)) - float(right_box.get("x", 0.0))) < 1e-6
            and abs(float(left_box.get("y", 0.0)) - float(right_box.get("y", 0.0))) < 1e-6
            and abs(float(left_box.get("z", 0.0)) - float(right_box.get("z", 0.0))) < 1e-6
        )
        if same_name or same_xyz:
            right_box = None

    return left_box, right_box


def _is_front_gear_name(name: str) -> bool:
    n = str(name).lower()
    return (
        ("front_landing_gear" in n)
        or ("landing_gear_front" in n)
        or ("nose_gear" in n)
        or ("front_gear" in n)
    )


def _strip_lr_suffix(name: str) -> str:
    n = str(name).strip().lower()
    n = re.sub(r"_(left|right)$", "", n)
    return n


def _make_front_center_box_from_pair(left_box: Dict[str, Any], right_box: Dict[str, Any]) -> Dict[str, Any]:
    wl = min(
        int(left_box.get("warning_level", -1)),
        int(right_box.get("warning_level", -1)),
    )
    return {
        "name": f"{_strip_lr_suffix(str(left_box.get('name', 'front_landing_gear')))}_center",
        "warning_level": wl,
        "x": 0.5 * (float(left_box.get("x", 0.0)) + float(right_box.get("x", 0.0))),
        "y": 0.5 * (float(left_box.get("y", 0.0)) + float(right_box.get("y", 0.0))),
        "z": 0.5 * (float(left_box.get("z", 0.0)) + float(right_box.get("z", 0.0))),
        "rx": 0.5 * (float(left_box.get("rx", 0.0)) + float(right_box.get("rx", 0.0))),
        "ry": 0.5 * (float(left_box.get("ry", 0.0)) + float(right_box.get("ry", 0.0))),
        "rz": 0.5 * (float(left_box.get("rz", 0.0)) + float(right_box.get("rz", 0.0))),
        "sx": 0.5 * (float(left_box.get("sx", 0.0)) + float(right_box.get("sx", 0.0))),
        "sy": 0.5 * (float(left_box.get("sy", 0.0)) + float(right_box.get("sy", 0.0))),
        "sz": 0.5 * (float(left_box.get("sz", 0.0)) + float(right_box.get("sz", 0.0))),
    }


def _build_front_center_from_lr_candidates(boxes: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    front_lr = [
        b for b in boxes
        if _is_front_gear_name(str(b.get("name", "")))
        and ("left" in str(b.get("name", "")).lower() or "right" in str(b.get("name", "")).lower())
    ]
    if not front_lr:
        return None
    by_base: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for b in front_lr:
        n = str(b.get("name", "")).lower()
        base = _strip_lr_suffix(n)
        slot = "left" if "left" in n else ("right" if "right" in n else "")
        if not slot:
            continue
        if base not in by_base:
            by_base[base] = {}
        by_base[base][slot] = b
    pair_boxes = [
        _make_front_center_box_from_pair(v["left"], v["right"])
        for v in by_base.values()
        if "left" in v and "right" in v
    ]
    if not pair_boxes:
        return None
    # Prefer centerline front box.
    return sorted(
        pair_boxes,
        key=lambda bb: abs(float(bb.get("y", 0.0))),
    )[0]


def _pick_front_box_with_fallback(
    boxes_target: List[Dict[str, Any]],
    boxes_all: List[Dict[str, Any]],
    target_level: int,
) -> Optional[Dict[str, Any]]:
    # 1) Try the already-selected target-level boxes first.
    direct = _pick_box_by_filters(boxes_target, list(WARNING_FRONT_GEAR_NAME_FILTERS))
    pair_center = _build_front_center_from_lr_candidates(boxes_target)
    if direct is not None and ("left" not in str(direct.get("name", "")).lower()) and ("right" not in str(direct.get("name", "")).lower()):
        return direct
    if pair_center is not None:
        return pair_center
    if direct is not None:
        return direct

    # 3) Fallback to lower warning levels (e.g., 777 front gears are often level 4/3).
    level_candidates = sorted(
        {
            int(b.get("warning_level", -1))
            for b in boxes_all
            if int(b.get("warning_level", -1)) >= 0
        },
        reverse=True,
    )
    level_candidates = [lvl for lvl in level_candidates if lvl < int(target_level)]

    for lvl in level_candidates:
        lvl_boxes = [b for b in boxes_all if int(b.get("warning_level", -1)) == int(lvl)]
        if not lvl_boxes:
            continue
        direct_lvl = _pick_box_by_filters(lvl_boxes, list(WARNING_FRONT_GEAR_NAME_FILTERS))
        pair_center_lvl = _build_front_center_from_lr_candidates(lvl_boxes)
        if direct_lvl is not None and ("left" not in str(direct_lvl.get("name", "")).lower()) and ("right" not in str(direct_lvl.get("name", "")).lower()):
            return direct_lvl
        if pair_center_lvl is not None:
            return pair_center_lvl
        if direct_lvl is not None:
            return direct_lvl

    return None


def _filters_include_front_box(csv_box_names: List[str]) -> bool:
    if not csv_box_names:
        return False
    return any(_is_front_gear_name(str(n)) for n in csv_box_names)


def _build_derived_nose_spec_from_scene_keypoints(
    names_scene: List[str],
    xyz_scene: np.ndarray,
    *,
    scale: float,
) -> Optional[Dict[str, Any]]:
    name_to_idx = {str(n): i for i, n in enumerate(names_scene)}
    need = ("plane_front_left_wheel_link", "plane_front_right_wheel_link")
    if any(k not in name_to_idx for k in need):
        return None
    fl = np.asarray(xyz_scene[name_to_idx[need[0]]], dtype=np.float64).reshape(3)
    fr = np.asarray(xyz_scene[name_to_idx[need[1]]], dtype=np.float64).reshape(3)
    if not (np.all(np.isfinite(fl)) and np.all(np.isfinite(fr))):
        return None
    center = 0.5 * (fl + fr)
    half = np.array([1.0, 1.0, 1.0], dtype=np.float64) * float(scale)  # merger default nose extent=(2,2,2)
    return {
        "source_name": "derived_nose_gear",
        "center_world": center,
        "half": half,
        "R_world_to_box": np.eye(3, dtype=np.float64),
    }


def _fit_rigid_local_to_world(local_pts: np.ndarray, world_pts: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if local_pts.ndim != 2 or world_pts.ndim != 2:
        return None
    # 2-point fit is underconstrained but still useful for engine-only scenes.
    if local_pts.shape != world_pts.shape or local_pts.shape[0] < 2 or local_pts.shape[1] != 3:
        return None

    p = np.asarray(local_pts, dtype=np.float64)
    q = np.asarray(world_pts, dtype=np.float64)
    cp = np.mean(p, axis=0)
    cq = np.mean(q, axis=0)
    p0 = p - cp.reshape(1, 3)
    q0 = q - cq.reshape(1, 3)

    H = p0.T @ q0
    U, _, Vt = np.linalg.svd(H)
    Rcol = Vt.T @ U.T
    if np.linalg.det(Rcol) < 0:
        Vt[-1, :] *= -1.0
        Rcol = Vt.T @ U.T

    t = cq - (Rcol @ cp)
    if not np.all(np.isfinite(Rcol)) or not np.all(np.isfinite(t)):
        return None
    return Rcol, t


def _normalize_name(name: str) -> str:
    return str(name).strip().lower()


def _find_observed_by_alias(
    kp_named: List[Tuple[str, np.ndarray]],
    aliases: List[str],
) -> Optional[Tuple[str, np.ndarray]]:
    if not kp_named:
        return None
    alias_l = [_normalize_name(a) for a in aliases]

    for nm, p in kp_named:
        n = _normalize_name(nm)
        if n in alias_l:
            return nm, p
    for nm, p in kp_named:
        n = _normalize_name(nm)
        if any(a in n for a in alias_l):
            return nm, p
    return None


def _target_warning_box_key_from_kp_name(name: str) -> Optional[str]:
    n = _normalize_name(name)
    if "engine_left" in n or "left_engine" in n:
        return "engine_left"
    if "engine_right" in n or "right_engine" in n:
        return "engine_right"
    if "front" in n or "nose" in n:
        return "front_landing_gear"
    return None


def _swap_engine_box_key_if_needed(box_key: str, *, swap_engine_lr: bool) -> str:
    if not swap_engine_lr:
        return str(box_key)
    if box_key == "engine_left":
        return "engine_right"
    if box_key == "engine_right":
        return "engine_left"
    return str(box_key)


def _build_checks_from_specs(
    kp_named: List[Tuple[str, np.ndarray]],
    specs: Dict[str, Dict[str, Any]],
    *,
    swap_engine_lr: bool,
) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    for nm, p in kp_named:
        target_box_key = _target_warning_box_key_from_kp_name(nm)
        if target_box_key is None:
            continue

        target_box_key = _swap_engine_box_key_if_needed(
            target_box_key,
            swap_engine_lr=swap_engine_lr,
        )
        if target_box_key not in specs:
            continue

        spec = specs[target_box_key]
        inside, local = _point_inside_warning_box(np.asarray(p, dtype=np.float64), spec)
        checks.append(
            {
                "kp_name": nm,
                "box_key": target_box_key,
                "inside": bool(inside),
                "abs_local": np.abs(local),
                "half": np.asarray(spec["half"], dtype=np.float64).reshape(3),
                "point": np.asarray(p, dtype=np.float64).reshape(3),
                "engine_lr_swapped": bool(swap_engine_lr),
            }
        )
    return checks


def _infer_scene_keypoint_names_from_csv(csv_path: Path) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    if not csv_path.exists() or not csv_path.is_file():
        return out

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        conf_cols = [c for c in fieldnames if str(c).startswith("conf_")]
        if not conf_cols:
            return out

        for row in reader:
            scene = str(row.get("unique_scene", "") or "").strip()
            if not scene:
                continue
            names: List[str] = []
            for c in conf_cols:
                v = str(row.get(c, "") or "").strip()
                if not v:
                    continue
                try:
                    float(v)
                except Exception:
                    continue
                names.append(c[len("conf_") :])
            if names:
                out[scene] = names
    return out


def _load_scene_keypoint_confidence_map(csv_path: Path) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    if not csv_path.exists() or not csv_path.is_file():
        return out

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        conf_cols = [c for c in fieldnames if str(c).startswith("conf_")]
        if not conf_cols:
            return out

        for row in reader:
            scene = str(row.get("unique_scene", "") or "").strip()
            if not scene:
                continue
            d: Dict[str, float] = {}
            for c in conf_cols:
                raw = str(row.get(c, "") or "").strip()
                if not raw:
                    continue
                try:
                    d[str(c)[len("conf_") :]] = float(raw)
                except Exception:
                    continue
            if d:
                out[scene] = d
    return out


def _parse_name_csv(raw: str) -> List[str]:
    return [str(x).strip() for x in str(raw or "").split(",") if str(x).strip()]


def _normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        raise ValueError("cannot normalize near-zero vector")
    return v / n


def _estimate_target_from_center_rotation_from_arrays(names: List[str], xyz: np.ndarray) -> Optional[np.ndarray]:
    req = [
        "left_wing_tip",
        "right_wing_tip",
        "plane_front_left_wheel_link",
        "plane_front_right_wheel_link",
        "plane_rear_left_wheel_link",
        "plane_rear_right_wheel_link",
    ]
    name_to_idx = {str(n): i for i, n in enumerate(names)}
    if any(k not in name_to_idx for k in req):
        return None
    left_tip = xyz[name_to_idx["left_wing_tip"]]
    right_tip = xyz[name_to_idx["right_wing_tip"]]
    front_left = xyz[name_to_idx["plane_front_left_wheel_link"]]
    front_right = xyz[name_to_idx["plane_front_right_wheel_link"]]
    rear_left = xyz[name_to_idx["plane_rear_left_wheel_link"]]
    rear_right = xyz[name_to_idx["plane_rear_right_wheel_link"]]

    y_axis = _normalize(left_tip - right_tip)
    front_mid = 0.5 * (front_left + front_right)
    rear_mid = 0.5 * (rear_left + rear_right)
    x_guess = front_mid - rear_mid
    x_axis = _normalize(x_guess - np.dot(x_guess, y_axis) * y_axis)
    z_axis = _normalize(np.cross(x_axis, y_axis))
    if z_axis[2] < 0.0:
        z_axis = -z_axis
    return np.column_stack([x_axis, y_axis, z_axis])


def _split_unique_scene(unique_scene: str) -> Tuple[str, Optional[str]]:
    s = str(unique_scene)
    m = re.match(r"^(.*)__(scene_\d+)$", s)
    if m:
        return str(m.group(1)), str(m.group(2))
    return _extract_bag_name_from_scene(s), None


def _resolve_h5_path_for_bag(
    bag_name: str,
    h5_root: str,
    h5_path_cache: Dict[str, Optional[Path]],
) -> Optional[Path]:
    if bag_name in h5_path_cache:
        return h5_path_cache[bag_name]

    root = Path(h5_root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        h5_path_cache[bag_name] = None
        return None

    token = _extract_aircraft_token_from_bag(bag_name)
    direct_candidates = [
        root / token / f"{bag_name}.h5",
        root / f"{bag_name}.h5",
    ]
    for p in direct_candidates:
        if p.exists() and p.is_file():
            h5_path_cache[bag_name] = p.resolve()
            return h5_path_cache[bag_name]

    hits = sorted(root.rglob(f"{bag_name}.h5"))
    if hits:
        h5_path_cache[bag_name] = hits[0].resolve()
        return h5_path_cache[bag_name]

    h5_path_cache[bag_name] = None
    return None


def _load_scene_keypoints_from_h5(
    unique_scene: str,
    warning_state: Dict[str, Any],
) -> Tuple[Optional[List[str]], Optional[np.ndarray], str]:
    cache: Dict[str, Tuple[Optional[List[str]], Optional[np.ndarray], str]] = warning_state["scene_keypoints_cache"]
    key = str(unique_scene)
    if key in cache:
        names, xyz, reason = cache[key]
        return names, xyz, reason

    bag_name, scene_name = _split_unique_scene(unique_scene)
    if not scene_name:
        out = (None, None, "scene name parse failed")
        cache[key] = out
        return out

    h5_path = _resolve_h5_path_for_bag(
        bag_name=bag_name,
        h5_root=str(warning_state["warning_h5_root"]),
        h5_path_cache=warning_state["h5_path_cache"],
    )
    if h5_path is None:
        out = (None, None, "h5 file not found for bag")
        cache[key] = out
        return out

    try:
        import h5py
    except Exception:
        out = (None, None, "h5py not available in this environment")
        cache[key] = out
        return out

    try:
        with h5py.File(h5_path, "r") as f:
            if scene_name not in f:
                out = (None, None, f"scene '{scene_name}' missing in {h5_path.name}")
                cache[key] = out
                return out
            grp = f[scene_name]
            if "keypoints" not in grp:
                out = (None, None, f"scene '{scene_name}' has no keypoints group")
                cache[key] = out
                return out
            kp_grp = grp["keypoints"]
            xyz = np.asarray(kp_grp["xyz"][()], dtype=np.float64)
            raw_names = kp_grp.get("names", None)
            names = [
                n.decode("utf-8") if isinstance(n, (bytes, bytearray)) else str(n)
                for n in (raw_names[()] if raw_names is not None else [])
            ]
    except Exception as e:
        out = (None, None, f"failed reading H5 scene keypoints: {e}")
        cache[key] = out
        return out

    if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.shape[0] == 0:
        out = (None, None, "h5 scene keypoints malformed")
        cache[key] = out
        return out

    ok_rows = np.all(np.isfinite(xyz), axis=1)
    xyz = xyz[ok_rows]
    if names:
        names = [names[i] for i, keep in enumerate(ok_rows) if keep]
    else:
        names = [f"k{i}" for i in range(xyz.shape[0])]

    out = (names, xyz, "")
    cache[key] = out
    return out


def _spec_from_box_with_local_pose(
    box: Dict[str, Any],
    *,
    R_world_to_local: np.ndarray,
    t_world: np.ndarray,
    scale: float,
) -> Dict[str, Any]:
    c_local = _box_center_local(box)
    c_world = c_local @ R_world_to_local.T + t_world.reshape(3)
    R_box_local = _euler_xyz_to_rotation_matrix(
        float(box.get("rx", 0.0)),
        float(box.get("ry", 0.0)),
        float(box.get("rz", 0.0)),
    )
    half = np.array(
        [
            max(0.25, 0.5 * abs(float(box.get("sx", 0.0)))),
            max(0.25, 0.5 * abs(float(box.get("sy", 0.0)))),
            max(0.25, 0.5 * abs(float(box.get("sz", 0.0)))),
        ],
        dtype=np.float64,
    ) * float(scale)
    return {
        "source_name": str(box.get("name", "")),
        "center_world": c_world,
        "half": half,
        "R_world_to_box": R_world_to_local @ R_box_local,
    }


def _extract_bag_name_from_scene(unique_scene: str) -> str:
    m = re.match(r"^(.*)__scene_\d+$", str(unique_scene))
    if m:
        return str(m.group(1))
    return str(unique_scene)


def _resolve_warning_yaml_path_for_scene(
    unique_scene: str,
    *,
    profile_map: Dict[str, Dict[str, str]],
    warning_yaml_column: str,
    warning_yaml_root: str,
    warning_yaml_relpath: str,
) -> Tuple[Optional[Path], Optional[Dict[str, str]]]:
    bag_name = _extract_bag_name_from_scene(unique_scene)
    row = _lookup_warning_profile_entry(profile_map, bag_name=bag_name)

    yaml_path: Optional[Path] = None
    if row is not None:
        ypath = str(row.get(warning_yaml_column, "") or "").strip()
        if ypath:
            p = Path(ypath).expanduser().resolve()
            if p.exists():
                yaml_path = p

    if yaml_path is None:
        yaml_path = _resolve_warning_yaml_for_bag(
            bag_name=bag_name,
            warning_yaml_root=warning_yaml_root,
            warning_yaml_relpath=warning_yaml_relpath,
        )
    return yaml_path, row


def _build_warning_specs_and_checks(
    unique_scene: str,
    kp_named: List[Tuple[str, np.ndarray]],
    *,
    warning_state: Dict[str, Any],
    warning_box_scale: float,
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]], str]:
    yaml_path, row = _resolve_warning_yaml_path_for_scene(
        unique_scene=unique_scene,
        profile_map=warning_state["profile_map"],
        warning_yaml_column=warning_state["warning_yaml_column"],
        warning_yaml_root=warning_state["warning_yaml_root"],
        warning_yaml_relpath=warning_state["warning_yaml_relpath"],
    )
    if yaml_path is None:
        return {}, [], "warning yaml not found"

    boxes_all = _load_yaml_crop_boxes(yaml_path, warning_state["yaml_cache"])
    if not boxes_all:
        return {}, [], "warning yaml has no crop_boxes"

    target_level = int(warning_state["warning_target_level"])
    boxes = [b for b in boxes_all if int(b.get("warning_level", -1)) == target_level]
    if not boxes:
        boxes = list(boxes_all)

    csv_box_names = _parse_csv_box_name_filters(row)
    if csv_box_names:
        csv_filters = [str(x).strip().lower() for x in csv_box_names if str(x).strip()]
        filtered = [
            b for b in boxes
            if any(f in str(b.get("name", "")).lower() for f in csv_filters)
        ]
        if filtered:
            boxes = filtered

    profile_has_front_filter = _filters_include_front_box(csv_box_names)
    front_box = _pick_front_box_with_fallback(
        boxes_target=boxes,
        boxes_all=boxes_all,
        target_level=target_level,
    )
    front_box_for_use = front_box
    # Keep merger parity: if profile does not ask for front gear boxes, do not
    # use YAML front fallback boxes for front checks.
    if bool(csv_box_names) and (not profile_has_front_filter):
        front_box_for_use = None
    left_box, right_box = _pick_engine_lr_boxes(boxes)
    scale = max(1e-6, float(warning_box_scale))
    specs: Dict[str, Dict[str, Any]] = {}
    scene_reason = ""

    # Preferred path: use scene keypoints from matching H5 (stable scene-wise transform).
    if bool(warning_state.get("use_scene_h5_transform", True)):
        names_scene, xyz_scene, scene_reason = _load_scene_keypoints_from_h5(
            unique_scene=unique_scene,
            warning_state=warning_state,
        )
        if names_scene is not None and xyz_scene is not None:
            name_to_idx = {str(n): i for i, n in enumerate(names_scene)}
            if "center" not in name_to_idx:
                scene_reason = "h5 scene keypoints missing 'center'"
            else:
                R_target = _estimate_target_from_center_rotation_from_arrays(names_scene, xyz_scene)
                if R_target is None:
                    scene_reason = "h5 scene keypoints missing wing/wheel links for orientation"
                else:
                    center_world = np.asarray(xyz_scene[name_to_idx["center"]], dtype=np.float64)
                    offset = np.asarray(
                        warning_state.get("warning_center_frame_offset", (0.0, 0.0, 0.0)),
                        dtype=np.float64,
                    ).reshape(3)
                    origin_world = center_world + (R_target @ offset)
                    # Match merger behavior: when profile filters do not include a front box
                    # (e.g., 777_300), prefer derived nose gear from wheel keypoints.
                    use_derived_nose = bool(csv_box_names) and (not profile_has_front_filter)
                    if use_derived_nose:
                        derived_spec = _build_derived_nose_spec_from_scene_keypoints(
                            names_scene=names_scene,
                            xyz_scene=xyz_scene,
                            scale=scale,
                        )
                        if derived_spec is not None:
                            specs["front_landing_gear"] = derived_spec
                    elif front_box_for_use is not None:
                        specs["front_landing_gear"] = _spec_from_box_with_local_pose(
                            front_box_for_use,
                            R_world_to_local=R_target,
                            t_world=origin_world,
                            scale=scale,
                        )
                    if left_box is not None:
                        specs["engine_left"] = _spec_from_box_with_local_pose(
                            left_box,
                            R_world_to_local=R_target,
                            t_world=origin_world,
                            scale=scale,
                        )
                    if right_box is not None:
                        specs["engine_right"] = _spec_from_box_with_local_pose(
                            right_box,
                            R_world_to_local=R_target,
                            t_world=origin_world,
                            scale=scale,
                        )
                    scene_reason = ""

    # Fallback path: estimate transform from observed keypoints in the PCD tail.
    if not specs:
        obs_front = _find_observed_by_alias(
            kp_named,
            ["front_landing_gear", "front_wheels_mid", "nose_gear", "front_gear"],
        )
        obs_left = _find_observed_by_alias(
            kp_named,
            ["engine_left_box_center", "engine_left", "left_engine"],
        )
        obs_right = _find_observed_by_alias(
            kp_named,
            ["engine_right_box_center", "engine_right", "right_engine"],
        )

        local_pts: List[np.ndarray] = []
        world_pts: List[np.ndarray] = []
        if front_box_for_use is not None and obs_front is not None:
            local_pts.append(_box_center_local(front_box_for_use))
            world_pts.append(np.asarray(obs_front[1], dtype=np.float64).reshape(3))
        if left_box is not None and obs_left is not None:
            local_pts.append(_box_center_local(left_box))
            world_pts.append(np.asarray(obs_left[1], dtype=np.float64).reshape(3))
        if right_box is not None and obs_right is not None:
            local_pts.append(_box_center_local(right_box))
            world_pts.append(np.asarray(obs_right[1], dtype=np.float64).reshape(3))

        if len(local_pts) < 2:
            if scene_reason:
                return {}, [], f"{scene_reason}; fallback needs at least 2 matched warning anchors (front/engine_left/engine_right)"
            return {}, [], "need at least 2 matched warning anchors (front/engine_left/engine_right)"

        fit = _fit_rigid_local_to_world(
            np.asarray(local_pts, dtype=np.float64),
            np.asarray(world_pts, dtype=np.float64),
        )
        if fit is None:
            if scene_reason:
                return {}, [], f"{scene_reason}; fallback failed to estimate warning-box transform"
            return {}, [], "failed to estimate warning-box transform"

        Rcol, t = fit
        if front_box_for_use is not None:
            specs["front_landing_gear"] = _spec_from_box_with_local_pose(
                front_box_for_use,
                R_world_to_local=Rcol,
                t_world=t,
                scale=scale,
            )
        if left_box is not None:
            specs["engine_left"] = _spec_from_box_with_local_pose(
                left_box,
                R_world_to_local=Rcol,
                t_world=t,
                scale=scale,
            )
        if right_box is not None:
            specs["engine_right"] = _spec_from_box_with_local_pose(
                right_box,
                R_world_to_local=Rcol,
                t_world=t,
                scale=scale,
            )

    checks_normal = _build_checks_from_specs(
        kp_named,
        specs,
        swap_engine_lr=False,
    )
    checks = checks_normal

    # Auto-recover left/right engine label mismatches by evaluating a swapped mapping.
    if "engine_left" in specs and "engine_right" in specs:
        checks_swapped = _build_checks_from_specs(
            kp_named,
            specs,
            swap_engine_lr=True,
        )
        inside_normal = sum(1 for c in checks_normal if bool(c["inside"]))
        inside_swapped = sum(1 for c in checks_swapped if bool(c["inside"]))
        if inside_swapped > inside_normal:
            checks = checks_swapped

    if not checks:
        return specs, checks, "no keypoints matched front/engine warning boxes"
    return specs, checks, ""


def _view_files(
    paths: List[Path],
    show_axes: bool,
    visualize: bool,
    kpt_count: int,
    kpt_radius: float,
    warning_check_enabled: bool,
    warning_pass_fail_enabled: bool,
    warning_keypoint_csv: Path,
    warning_profile_csv: str,
    warning_yaml_column: str,
    warning_yaml_root: str,
    warning_yaml_relpath: str,
    warning_target_level: int,
    warning_box_scale: float,
    warning_fallback_kp_names: List[str],
    warning_h5_root: str,
    use_scene_h5_transform: bool,
    warning_kp_passfail_csv: Optional[Path] = None,
    warning_conf_threshold: Optional[float] = None,
    infer_kpt_count_from_csv: bool = False,
    show_kp_spheres: bool = True,
    tail_points_label: str = "keypoints",
    engine_region_root: Optional[Path] = None,
    show_engine_region_points: bool = True,
    use_engine_region_ratio_for_passfail: bool = False,
    engine_region_inside_ratio_thr: float = 0.80,
    visualize_failed_scenes_only: bool = False,
    min_warning_fail_kp_to_visualize: int = 0,
) -> None:
    o3d = None
    if bool(visualize):
        try:
            import open3d as o3d
        except Exception as e:
            raise RuntimeError(
                "open3d is required for visualization. Install with `pip install open3d` "
                "or run with `--no-vis`."
            ) from e
    else:
        try:
            import open3d as o3d
        except Exception:
            o3d = None

    scene_kp_name_map: Dict[str, List[str]] = {}
    scene_kp_conf_map: Dict[str, Dict[str, float]] = {}
    warning_state: Dict[str, Any] = {
        "profile_map": {},
        "yaml_cache": {},
        "h5_path_cache": {},
        "scene_keypoints_cache": {},
        "warning_yaml_column": str(warning_yaml_column),
        "warning_yaml_root": str(warning_yaml_root),
        "warning_yaml_relpath": str(warning_yaml_relpath),
        "warning_target_level": int(warning_target_level),
        "warning_center_frame_offset": tuple(WARNING_CENTER_FRAME_OFFSET),
        "warning_h5_root": str(warning_h5_root),
        "use_scene_h5_transform": bool(use_scene_h5_transform),
    }

    if warning_check_enabled:
        if warning_keypoint_csv.exists() and warning_keypoint_csv.is_file():
            scene_kp_name_map = _infer_scene_keypoint_names_from_csv(warning_keypoint_csv)
            scene_kp_conf_map = _load_scene_keypoint_confidence_map(warning_keypoint_csv)
            print(f"[warning-check] scene keypoint CSV: {warning_keypoint_csv} (rows={len(scene_kp_name_map)})")
        else:
            print(f"[warning-check] keypoint CSV not found: {warning_keypoint_csv} (checks may be unavailable)")

        warning_state["profile_map"] = _load_warning_profile_map(str(warning_profile_csv or ""))
        if bool(use_scene_h5_transform):
            root = Path(str(warning_h5_root)).expanduser().resolve()
            print(f"[warning-check] scene H5 transform: enabled (root={root})")
        else:
            print("[warning-check] scene H5 transform: disabled (using keypoint-fit fallback only)")

    engine_region_root_path: Optional[Path] = None
    if engine_region_root is not None:
        engine_region_root_path = Path(str(engine_region_root)).expanduser().resolve()

    processed = 0
    shown = 0
    vis_skipped_by_fail_filter = 0
    vis_skipped_by_scene_result = 0
    empty = 0
    pf_evaluated = 0
    pf_pass = 0
    pf_fail = 0
    pf_fail_reasons: Dict[str, int] = {}
    pf_pass_checks = 0
    pf_pass_inside = 0
    pf_pass_by_inside: Dict[int, int] = {}
    ck_total = 0
    ck_inside = 0
    ck_outside = 0
    ratio_mode_sides_checked = 0
    ratio_mode_sides_pass = 0
    ratio_mode_warning_points_total = 0
    ratio_mode_warning_points_matched = 0
    kp_pf_rows: List[Dict[str, Any]] = []
    ratio_thr = 0.80
    try:
        ratio_thr = float(engine_region_inside_ratio_thr)
    except Exception:
        ratio_thr = 0.80
    ratio_thr = float(np.clip(ratio_thr, 0.0, 1.0))
    conf_thr_use: Optional[float] = None
    if warning_conf_threshold is not None:
        try:
            v = float(warning_conf_threshold)
            if v >= 0.0:
                conf_thr_use = v
        except Exception:
            conf_thr_use = None

    if not bool(visualize):
        print("[viewer] visualization: disabled (--no-vis). Running checks headless.")
        if o3d is None:
            print("[viewer] open3d not found; using ASCII .pcd/.ply fallback reader.")
    elif bool(visualize_failed_scenes_only) and not (
        bool(warning_check_enabled) and bool(warning_pass_fail_enabled)
    ):
        print(
            "[viewer] visualize_failed_scenes_only requested but warning pass/fail is disabled; "
            "showing all scenes."
        )

    for i, path in enumerate(paths, 1):
        try:
            xyz = _read_xyz(path, o3d=o3d)
        except Exception as e:
            empty += 1
            print(f"[{i}/{len(paths)}] {path.name} -> read failed: {e}")
            continue
        if xyz.size == 0:
            empty += 1
            print(f"[{i}/{len(paths)}] {path.name} -> empty point cloud")
            continue

        unique_scene = str(path.stem)
        processed += 1

        print(f"[{i}/{len(paths)}] {path}")
        _print_cloud_summary(path, xyz)

        kpt_count_scene = int(kpt_count)
        if bool(infer_kpt_count_from_csv):
            kpt_count_scene = int(len(scene_kp_name_map.get(unique_scene, [])))
        xyz_main, xyz_kp = _split_main_and_keypoints(xyz, kpt_count=kpt_count_scene)
        if xyz_kp.shape[0] > 0:
            print(f"  {str(tail_points_label)}(from tail)={xyz_kp.shape[0]}")

        kp_names_csv = scene_kp_name_map.get(unique_scene, []) if warning_check_enabled else []
        kp_named: List[Tuple[str, np.ndarray]] = []
        kp_name_note = ""

        if kp_names_csv:
            n_map = min(int(xyz_kp.shape[0]), len(kp_names_csv))
            for j in range(n_map):
                kp_named.append((str(kp_names_csv[j]), xyz_kp[j]))
            for j in range(n_map, int(xyz_kp.shape[0])):
                if j < len(warning_fallback_kp_names):
                    kp_named.append((str(warning_fallback_kp_names[j]), xyz_kp[j]))
                else:
                    kp_named.append((f"K{j}", xyz_kp[j]))
            if len(kp_names_csv) != int(xyz_kp.shape[0]):
                kp_name_note = (
                    f"CSV keypoint-name count ({len(kp_names_csv)}) != tail keypoint count ({int(xyz_kp.shape[0])}); "
                    "used partial CSV names + fallback names."
                )
        elif warning_check_enabled and int(xyz_kp.shape[0]) > 0:
            if int(xyz_kp.shape[0]) <= len(warning_fallback_kp_names):
                kp_named = [
                    (str(warning_fallback_kp_names[j]), xyz_kp[j])
                    for j in range(int(xyz_kp.shape[0]))
                ]
                kp_name_note = "scene missing in keypoint CSV; using fallback keypoint names."
            else:
                kp_named = [(f"K{j}", xyz_kp[j]) for j in range(int(xyz_kp.shape[0]))]
                kp_name_note = (
                    "scene missing in keypoint CSV and fallback list shorter than tail keypoint count; "
                    "using generic K0.. names."
                )
        else:
            kp_named = [(f"K{j}", xyz_kp[j]) for j in range(int(xyz_kp.shape[0]))]

        if kp_name_note:
            print(f"  warning-check: {kp_name_note}")

        left_xyz = np.empty((0, 3), dtype=np.float64)
        right_xyz = np.empty((0, 3), dtype=np.float64)
        front_xyz = np.empty((0, 3), dtype=np.float64)
        need_engine_regions = bool(
            engine_region_root_path is not None
            and (
                (bool(visualize) and bool(show_engine_region_points))
                or (bool(warning_pass_fail_enabled) and bool(use_engine_region_ratio_for_passfail))
            )
        )
        if need_engine_regions and engine_region_root_path is not None:
            left_fp = engine_region_root_path / "engine_left" / f"{unique_scene}.pcd"
            right_fp = engine_region_root_path / "engine_right" / f"{unique_scene}.pcd"
            front_fp = engine_region_root_path / "front_gear" / f"{unique_scene}.pcd"
            if left_fp.exists() and left_fp.is_file():
                try:
                    left_xyz = _read_xyz(left_fp, o3d=o3d)
                except Exception as e:
                    print(f"  [warn] failed loading engine_left region {left_fp.name}: {e}")
            if right_fp.exists() and right_fp.is_file():
                try:
                    right_xyz = _read_xyz(right_fp, o3d=o3d)
                except Exception as e:
                    print(f"  [warn] failed loading engine_right region {right_fp.name}: {e}")
            if front_fp.exists() and front_fp.is_file():
                try:
                    front_xyz = _read_xyz(front_fp, o3d=o3d)
                except Exception as e:
                    print(f"  [warn] failed loading front_gear region {front_fp.name}: {e}")

        warning_specs: Dict[str, Dict[str, Any]] = {}
        warning_checks: List[Dict[str, Any]] = []
        warning_reason = ""
        scene_pf_label = ""
        scene_pf_reason = ""
        if warning_check_enabled and (kp_named or warning_pass_fail_enabled):
            warning_specs, warning_checks, warning_reason = _build_warning_specs_and_checks(
                unique_scene=unique_scene,
                kp_named=kp_named,
                warning_state=warning_state,
                warning_box_scale=warning_box_scale,
            )
            print_anchor_checks = not (
                bool(warning_pass_fail_enabled) and bool(use_engine_region_ratio_for_passfail)
            )
            if warning_checks and print_anchor_checks:
                print("  warning-check:")
                if any(bool(c.get("engine_lr_swapped", False)) for c in warning_checks):
                    print("    - engine mapping auto-swap applied (left/right)")
                for ck in warning_checks:
                    a = np.asarray(ck["abs_local"], dtype=np.float64).reshape(3)
                    h = np.asarray(ck["half"], dtype=np.float64).reshape(3)
                    status = "INSIDE" if bool(ck["inside"]) else "OUTSIDE"
                    print(
                        f"    - {ck['kp_name']} vs {ck['box_key']}: {status} "
                        f"| abs(local)=({a[0]:.3f},{a[1]:.3f},{a[2]:.3f}) "
                        f"half=({h[0]:.3f},{h[1]:.3f},{h[2]:.3f})"
                    )
            elif warning_reason and print_anchor_checks:
                print(f"  warning-check: {warning_reason}")

            if warning_pass_fail_enabled:
                pf_evaluated += 1
                if bool(use_engine_region_ratio_for_passfail):
                    if not warning_specs:
                        scene_pf_label = "FAIL"
                        scene_pf_reason = str(warning_reason or "warning specs unavailable")
                        pf_fail += 1
                        pf_fail_reasons[scene_pf_reason] = int(pf_fail_reasons.get(scene_pf_reason, 0)) + 1
                    elif engine_region_root_path is None:
                        scene_pf_label = "FAIL"
                        scene_pf_reason = "engine region root not configured"
                        pf_fail += 1
                        pf_fail_reasons[scene_pf_reason] = int(pf_fail_reasons.get(scene_pf_reason, 0)) + 1
                    else:
                        print(
                            "  warning-check (warning-box coverage by detected engine regions, "
                            f"thr>={ratio_thr:.3f}):"
                        )
                        failed_sides: List[str] = []
                        checked_sides = 0
                        det_points_by_side: Dict[str, np.ndarray] = {}
                        for side_key, side_pts in (
                            ("engine_left", left_xyz),
                            ("engine_right", right_xyz),
                        ):
                            pts_det = np.asarray(side_pts, dtype=np.float64).reshape(-1, 3)
                            finite_det = np.all(np.isfinite(pts_det), axis=1)
                            det_points_by_side[side_key] = pts_det[finite_det]

                        warn_points_by_side: Dict[str, np.ndarray] = {}
                        for side_key in ("engine_left", "engine_right"):
                            if side_key not in warning_specs:
                                warn_points_by_side[side_key] = np.empty((0, 3), dtype=np.float64)
                                continue
                            pts_warn = np.asarray(
                                xyz_main[_inside_mask_warning_box(xyz_main, warning_specs[side_key])],
                                dtype=np.float64,
                            ).reshape(-1, 3)
                            finite_warn = np.all(np.isfinite(pts_warn), axis=1)
                            warn_points_by_side[side_key] = pts_warn[finite_warn]

                        def _coverage_ratio(det_side: str, warn_side: str) -> float:
                            pts_det = det_points_by_side.get(det_side, np.empty((0, 3), dtype=np.float64))
                            pts_warn = warn_points_by_side.get(warn_side, np.empty((0, 3), dtype=np.float64))
                            if pts_det.size == 0 or pts_warn.size == 0:
                                return -1.0
                            overlap_n, warn_n, _det_n = _count_quantized_point_overlap(
                                pts_warn, pts_det, quant_scale=1000.0
                            )
                            return float(overlap_n) / float(max(1, warn_n))

                        det_for_warning_side: Dict[str, Optional[str]] = {
                            "engine_left": "engine_left",
                            "engine_right": "engine_right",
                        }
                        has_left_det = det_points_by_side["engine_left"].size > 0
                        has_right_det = det_points_by_side["engine_right"].size > 0

                        if has_left_det and has_right_det:
                            lr_overlap_n, left_u_n, right_u_n = _count_quantized_point_overlap(
                                det_points_by_side["engine_left"],
                                det_points_by_side["engine_right"],
                                quant_scale=1000.0,
                            )
                            denom = max(1, min(int(left_u_n), int(right_u_n)))
                            lr_overlap_ratio = float(lr_overlap_n) / float(denom)
                            if lr_overlap_ratio >= float(ENGINE_LR_OVERLAP_AS_ONE_RATIO_THR):
                                det_points_by_side["engine_left"] = np.concatenate(
                                    [
                                        det_points_by_side["engine_left"],
                                        det_points_by_side["engine_right"],
                                    ],
                                    axis=0,
                                )
                                det_points_by_side["engine_right"] = np.empty(
                                    (0, 3), dtype=np.float64
                                )
                                print(
                                    "    - engine-left/right detected regions overlap; "
                                    f"treating as one region (overlap={lr_overlap_n}/{denom}, "
                                    f"ratio={lr_overlap_ratio:.3f})"
                                )
                                has_left_det = True
                                has_right_det = False

                        if has_left_det and has_right_det:
                            score_normal = _coverage_ratio("engine_left", "engine_left") + _coverage_ratio(
                                "engine_right", "engine_right"
                            )
                            score_swapped = _coverage_ratio("engine_left", "engine_right") + _coverage_ratio(
                                "engine_right", "engine_left"
                            )
                            if score_swapped > score_normal:
                                det_for_warning_side["engine_left"] = "engine_right"
                                det_for_warning_side["engine_right"] = "engine_left"
                                print("    - engine mapping auto-swap applied (coverage-based)")
                        elif has_left_det and (not has_right_det):
                            to_left = _coverage_ratio("engine_left", "engine_left")
                            to_right = _coverage_ratio("engine_left", "engine_right")
                            if to_right > to_left:
                                det_for_warning_side["engine_left"] = None
                                det_for_warning_side["engine_right"] = "engine_left"
                                print("    - single detected engine region mapped to engine_right (higher coverage)")
                            else:
                                det_for_warning_side["engine_left"] = "engine_left"
                                det_for_warning_side["engine_right"] = None
                                print("    - single detected engine region mapped to engine_left (higher coverage)")
                        elif has_right_det and (not has_left_det):
                            to_left = _coverage_ratio("engine_right", "engine_left")
                            to_right = _coverage_ratio("engine_right", "engine_right")
                            if to_left > to_right:
                                det_for_warning_side["engine_left"] = "engine_right"
                                det_for_warning_side["engine_right"] = None
                                print("    - single detected engine region mapped to engine_left (higher coverage)")
                            else:
                                det_for_warning_side["engine_left"] = None
                                det_for_warning_side["engine_right"] = "engine_right"
                                print("    - single detected engine region mapped to engine_right (higher coverage)")
                        else:
                            det_for_warning_side["engine_left"] = None
                            det_for_warning_side["engine_right"] = None

                        for side_key in ("engine_left", "engine_right"):
                            if side_key not in warning_specs:
                                print(f"    - {side_key}: warning box unavailable (skipped)")
                                continue

                            pts_warn = warn_points_by_side.get(side_key, np.empty((0, 3), dtype=np.float64))
                            if pts_warn.size == 0:
                                print(f"    - {side_key}: no warning-box points from cloud (skipped)")
                                continue

                            det_side = det_for_warning_side.get(side_key, None)
                            if not det_side:
                                print(f"    - {side_key}: no detected engine region points (skipped)")
                                continue

                            pts_det = det_points_by_side.get(det_side, np.empty((0, 3), dtype=np.float64))
                            if pts_det.size == 0:
                                print(f"    - {side_key}: no detected engine region points (skipped)")
                                continue

                            overlap_n, warn_n, det_n = _count_quantized_point_overlap(
                                pts_warn,
                                pts_det,
                                quant_scale=1000.0,
                            )
                            checked_sides += 1
                            ratio = float(overlap_n) / float(max(1, warn_n))
                            ratio_mode_sides_checked += 1
                            ratio_mode_warning_points_total += warn_n
                            ratio_mode_warning_points_matched += overlap_n
                            status = "PASS"
                            if ratio < ratio_thr:
                                failed_sides.append(side_key)
                                status = "FAIL"
                            else:
                                ratio_mode_sides_pass += 1
                            src_note = "" if det_side == side_key else f" source={det_side}"
                            print(
                                f"    - {side_key}:{src_note} matched_warning={overlap_n}/{warn_n} "
                                f"(ratio={ratio:.3f}) detected_unique={det_n} -> {status}"
                            )

                        if checked_sides <= 0:
                            scene_pf_label = "FAIL"
                            scene_pf_reason = "no detected engine region points for warning check"
                            pf_fail += 1
                            pf_fail_reasons[scene_pf_reason] = int(pf_fail_reasons.get(scene_pf_reason, 0)) + 1
                        elif failed_sides:
                            scene_pf_label = "FAIL"
                            scene_pf_reason = (
                                f"below warning-box coverage threshold ({ratio_thr:.2f}): "
                                + ",".join(failed_sides)
                            )
                            pf_fail += 1
                            pf_fail_reasons[scene_pf_reason] = int(pf_fail_reasons.get(scene_pf_reason, 0)) + 1
                        else:
                            scene_pf_label = "PASS"
                            pf_pass += 1
                else:
                    if warning_checks:
                        inside_n = sum(1 for c in warning_checks if bool(c["inside"]))
                        outside_n = int(len(warning_checks) - inside_n)
                        ck_total += int(len(warning_checks))
                        ck_inside += int(inside_n)
                        ck_outside += int(outside_n)
                        if outside_n == 0:
                            scene_pf_label = "PASS"
                            pf_pass += 1
                            pf_pass_checks += int(len(warning_checks))
                            pf_pass_inside += int(inside_n)
                            pf_pass_by_inside[int(inside_n)] = int(pf_pass_by_inside.get(int(inside_n), 0)) + 1
                        else:
                            scene_pf_label = "FAIL"
                            scene_pf_reason = f"{outside_n} outside warning box"
                            pf_fail += 1
                            pf_fail_reasons[scene_pf_reason] = int(pf_fail_reasons.get(scene_pf_reason, 0)) + 1
                    else:
                        scene_pf_label = "FAIL"
                        scene_pf_reason = str(warning_reason or "no warning-box checks produced")
                        pf_fail += 1
                        pf_fail_reasons[scene_pf_reason] = int(pf_fail_reasons.get(scene_pf_reason, 0)) + 1
                if scene_pf_reason:
                    print(f"  warning-result: {scene_pf_label} ({scene_pf_reason})")
                else:
                    print(f"  warning-result: {scene_pf_label}")

            if warning_checks:
                scene_conf = scene_kp_conf_map.get(unique_scene, {})
                scene_result = scene_pf_label
                if not scene_result:
                    outside_n = sum(1 for c in warning_checks if not bool(c["inside"]))
                    scene_result = "PASS" if outside_n == 0 else "FAIL"
                for ck in warning_checks:
                    kp_name = str(ck["kp_name"])
                    conf_v = scene_conf.get(kp_name, None)
                    include_by_thr = int(
                        conf_v is not None and (conf_thr_use is None or float(conf_v) >= float(conf_thr_use))
                    )
                    a = np.asarray(ck["abs_local"], dtype=np.float64).reshape(3)
                    h = np.asarray(ck["half"], dtype=np.float64).reshape(3)
                    kp_pf_rows.append(
                        {
                            "unique_scene": unique_scene,
                            "pcd_file": path.name,
                            "kp_name": kp_name,
                            "box_key": str(ck["box_key"]),
                            "check_status": "PASS" if bool(ck["inside"]) else "FAIL",
                            "scene_warning_result": str(scene_result),
                            "confidence": conf_v,
                            "has_confidence": 1 if conf_v is not None else 0,
                            "included_in_conf_threshold_stats": include_by_thr,
                            "confidence_threshold": conf_thr_use,
                            "abs_local_x": float(a[0]),
                            "abs_local_y": float(a[1]),
                            "abs_local_z": float(a[2]),
                            "half_x": float(h[0]),
                            "half_y": float(h[1]),
                            "half_z": float(h[2]),
                            "engine_lr_swapped": 1 if bool(ck.get("engine_lr_swapped", False)) else 0,
                        }
                    )
            elif scene_pf_label:
                kp_pf_rows.append(
                    {
                        "unique_scene": unique_scene,
                        "pcd_file": path.name,
                        "kp_name": "__scene__",
                        "box_key": "scene_summary",
                        "check_status": str(scene_pf_label),
                        "scene_warning_result": str(scene_pf_label),
                        "confidence": None,
                        "has_confidence": 0,
                        "included_in_conf_threshold_stats": 0,
                        "confidence_threshold": conf_thr_use,
                        "abs_local_x": 0.0,
                        "abs_local_y": 0.0,
                        "abs_local_z": 0.0,
                        "half_x": 0.0,
                        "half_y": 0.0,
                        "half_z": 0.0,
                        "engine_lr_swapped": 0,
                    }
                )

        scene_fail_kp_n = sum(1 for c in warning_checks if not bool(c.get("inside", False)))

        if not bool(visualize):
            continue
        if (
            bool(visualize_failed_scenes_only)
            and bool(warning_check_enabled)
            and bool(warning_pass_fail_enabled)
            and str(scene_pf_label).upper() != "FAIL"
        ):
            vis_skipped_by_scene_result += 1
            continue
        if (
            bool(warning_check_enabled)
            and int(min_warning_fail_kp_to_visualize) > 0
            and int(scene_fail_kp_n) < int(min_warning_fail_kp_to_visualize)
        ):
            vis_skipped_by_fail_filter += 1
            continue

        geoms: List[object] = []

        pcd_main = o3d.geometry.PointCloud()
        pcd_main.points = o3d.utility.Vector3dVector(xyz_main.astype(np.float64))
        pcd_main.paint_uniform_color([1.0, 0.1, 0.1])
        geoms.append(pcd_main)

        if warning_specs:
            for box_key, spec in sorted(warning_specs.items()):
                corners = _oriented_box_corners(
                    center_world=np.asarray(spec["center_world"], dtype=np.float64),
                    half=np.asarray(spec["half"], dtype=np.float64),
                    R_world_to_box=np.asarray(spec["R_world_to_box"], dtype=np.float64),
                )
                ls = o3d.geometry.LineSet()
                ls.points = o3d.utility.Vector3dVector(corners)
                ls.lines = o3d.utility.Vector2iVector(np.asarray(_warning_edges(), dtype=np.int32))
                col = _warning_color_rgb01(box_key).reshape(1, 3)
                ls.colors = o3d.utility.Vector3dVector(np.tile(col, (len(_warning_edges()), 1)))
                geoms.append(ls)

        if bool(show_engine_region_points) and engine_region_root_path is not None:
            if left_xyz.size > 0:
                pcd_left = o3d.geometry.PointCloud()
                pcd_left.points = o3d.utility.Vector3dVector(left_xyz.astype(np.float64))
                pcd_left.paint_uniform_color(_warning_color_rgb01("engine_left").tolist())
                geoms.append(pcd_left)
            if right_xyz.size > 0:
                pcd_right = o3d.geometry.PointCloud()
                pcd_right.points = o3d.utility.Vector3dVector(right_xyz.astype(np.float64))
                pcd_right.paint_uniform_color(_warning_color_rgb01("engine_right").tolist())
                geoms.append(pcd_right)
            if front_xyz.size > 0:
                pcd_front = o3d.geometry.PointCloud()
                pcd_front.points = o3d.utility.Vector3dVector(front_xyz.astype(np.float64))
                pcd_front.paint_uniform_color(_warning_color_rgb01("front_gear").tolist())
                geoms.append(pcd_front)
            if left_xyz.size > 0 or right_xyz.size > 0 or front_xyz.size > 0:
                print(
                    f"  engine_regions_3d: "
                    f"left={int(left_xyz.shape[0])} right={int(right_xyz.shape[0])} "
                    f"front={int(front_xyz.shape[0])}"
                )

        check_by_name = {str(c["kp_name"]): bool(c["inside"]) for c in warning_checks}
        if bool(show_kp_spheres):
            for name, p in kp_named:
                sph = o3d.geometry.TriangleMesh.create_sphere(radius=float(max(0.01, kpt_radius)))
                sph.compute_vertex_normals()
                box_key = _target_warning_box_key_from_kp_name(str(name))
                if box_key is not None:
                    kp_col = _warning_color_rgb01(box_key)
                else:
                    kp_col = np.array([0.1, 0.45, 1.0], dtype=np.float64)
                # Keep role color hue; dim when outside warning box.
                if name in check_by_name and (not bool(check_by_name[name])):
                    kp_col = np.clip(kp_col * 0.55, 0.0, 1.0)
                sph.paint_uniform_color(kp_col.tolist())
                sph.translate(np.asarray(p, dtype=np.float64).reshape(3), relative=False)
                geoms.append(sph)

        if show_axes:
            extent = np.ptp(xyz, axis=0)
            axis_size = float(max(0.1, np.max(extent) * 0.12))
            geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size))

        o3d.visualization.draw_geometries(
            geoms,
            window_name=f"PCD Viewer: {path.name}",
            width=1400,
            height=900,
        )
        shown += 1

    print(f"\n[summary] requested={len(paths)} processed={processed} shown={shown} empty={empty}")
    if bool(visualize) and int(min_warning_fail_kp_to_visualize) > 0:
        print(
            "[summary] visualize fail-kp filter: "
            f"min_fail_kp={int(min_warning_fail_kp_to_visualize)} "
            f"skipped={int(vis_skipped_by_fail_filter)}"
        )
    if bool(visualize) and bool(visualize_failed_scenes_only):
        print(
            "[summary] visualize failed-only filter: "
            f"skipped_non_fail={int(vis_skipped_by_scene_result)}"
        )
    if warning_pass_fail_enabled and warning_check_enabled:
        print(
            f"[summary] warning pass/fail: evaluated={pf_evaluated} pass={pf_pass} fail={pf_fail}"
        )
        if bool(use_engine_region_ratio_for_passfail):
            ratio_global = (
                float(ratio_mode_warning_points_matched) / float(ratio_mode_warning_points_total)
                if ratio_mode_warning_points_total > 0
                else float("nan")
            )
            ratio_global_txt = f"{ratio_global:.3f}" if ratio_mode_warning_points_total > 0 else "n/a"
            print(
                "[summary] warning-box coverage checks: "
                f"thr>={ratio_thr:.3f} sides_checked={ratio_mode_sides_checked} "
                f"sides_pass={ratio_mode_sides_pass} "
                f"matched_warning_points={ratio_mode_warning_points_matched}/{ratio_mode_warning_points_total} "
                f"global_warning_coverage_ratio={ratio_global_txt}"
            )
        else:
            print(
                f"[summary] warning checks: total={ck_total} inside={ck_inside} outside={ck_outside}"
            )
            print(
                f"[summary] pass-inside checks: inside={pf_pass_inside} total_in_pass={pf_pass_checks}"
            )
            if pf_pass_by_inside:
                for inside_n, scene_n in sorted(pf_pass_by_inside.items()):
                    print(f"[summary] pass-scenes: {inside_n} inside -> {scene_n}")
        if pf_fail_reasons:
            for reason, count in sorted(pf_fail_reasons.items(), key=lambda kv: (-kv[1], kv[0])):
                print(f"[summary] fail-reason: {reason} -> {count}")

    if warning_check_enabled and kp_pf_rows:
        if bool(use_engine_region_ratio_for_passfail):
            print(
                "[summary] note: keypoint confidence stats below are anchor-based "
                "and separate from engine-region ratio pass/fail."
            )
        below_thr_n = 0
        no_conf_n = 0
        if conf_thr_use is not None:
            for r in kp_pf_rows:
                conf_v = r.get("confidence", None)
                if conf_v is None:
                    no_conf_n += 1
                    continue
                if float(conf_v) < float(conf_thr_use):
                    below_thr_n += 1

        pass_conf = [
            float(r["confidence"])
            for r in kp_pf_rows
            if str(r.get("check_status")) == "PASS"
            and r.get("confidence", None) is not None
            and int(r.get("included_in_conf_threshold_stats", 0)) == 1
        ]
        fail_conf = [
            float(r["confidence"])
            for r in kp_pf_rows
            if str(r.get("check_status")) == "FAIL"
            and r.get("confidence", None) is not None
            and int(r.get("included_in_conf_threshold_stats", 0)) == 1
        ]
        if pass_conf or fail_conf:
            pass_mean = (sum(pass_conf) / len(pass_conf)) if pass_conf else float("nan")
            fail_mean = (sum(fail_conf) / len(fail_conf)) if fail_conf else float("nan")
            pass_mean_txt = f"{pass_mean:.4f}" if pass_conf else "n/a"
            fail_mean_txt = f"{fail_mean:.4f}" if fail_conf else "n/a"
            if conf_thr_use is not None:
                print(
                    "[summary] keypoint confidence by check: "
                    f"thr>={float(conf_thr_use):.3f} pass_n={len(pass_conf)} fail_n={len(fail_conf)} "
                    f"pass_mean={pass_mean_txt} fail_mean={fail_mean_txt}"
                )
            else:
                print(
                    "[summary] keypoint confidence by check: "
                    f"pass_n={len(pass_conf)} fail_n={len(fail_conf)} "
                    f"pass_mean={pass_mean_txt} fail_mean={fail_mean_txt}"
                )
        elif conf_thr_use is not None:
            print(
                "[summary] keypoint confidence by check: "
                f"thr>={float(conf_thr_use):.3f} pass_n=0 fail_n=0"
            )

        if conf_thr_use is not None:
            print(
                "[summary] keypoint confidence threshold filter: "
                f"below_thr={below_thr_n} missing_conf={no_conf_n}"
            )

        out_csv = warning_kp_passfail_csv
        if out_csv is None:
            out_csv = warning_keypoint_csv.parent / "keypoint_pass_fail_confidence.csv"
        out_csv = Path(out_csv).expanduser().resolve()
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow(
                [
                    "unique_scene",
                    "pcd_file",
                    "kp_name",
                    "box_key",
                    "check_status",
                    "scene_warning_result",
                    "confidence",
                    "has_confidence",
                    "included_in_conf_threshold_stats",
                    "confidence_threshold",
                    "abs_local_x",
                    "abs_local_y",
                    "abs_local_z",
                    "half_x",
                    "half_y",
                    "half_z",
                    "engine_lr_swapped",
                ]
            )
            for r in kp_pf_rows:
                conf_v = r.get("confidence", None)
                writer.writerow(
                    [
                        r.get("unique_scene", ""),
                        r.get("pcd_file", ""),
                        r.get("kp_name", ""),
                        r.get("box_key", ""),
                        r.get("check_status", ""),
                        r.get("scene_warning_result", ""),
                        "" if conf_v is None else f"{float(conf_v):.6f}",
                        int(r.get("has_confidence", 0)),
                        int(r.get("included_in_conf_threshold_stats", 0)),
                        ""
                        if r.get("confidence_threshold", None) is None
                        else f"{float(r.get('confidence_threshold')):.6f}",
                        f"{float(r.get('abs_local_x', 0.0)):.6f}",
                        f"{float(r.get('abs_local_y', 0.0)):.6f}",
                        f"{float(r.get('abs_local_z', 0.0)):.6f}",
                        f"{float(r.get('half_x', 0.0)):.6f}",
                        f"{float(r.get('half_y', 0.0)):.6f}",
                        f"{float(r.get('half_z', 0.0)):.6f}",
                        int(r.get("engine_lr_swapped", 0)),
                    ]
                )
        print(f"[summary] keypoint pass/fail confidence CSV: {out_csv}")


def parse_args():
    ap = argparse.ArgumentParser(description="View .pcd/.ply files from a directory one-by-one.")
    ap.add_argument("--root", type=str, default="/home/femi/yolo_pose_dataset_creation/pcd_from_yolo", help="Directory containing point-cloud files")
    ap.add_argument("--file", type=str, default=None, help="View only one specific file")
    ap.add_argument("--recursive", action="store_true", help="Search files recursively under --root")
    ap.add_argument("--pattern", type=str, default="", help="Filter by substring in filename")
    ap.add_argument("--exts", type=str, default="pcd,ply", help="Comma-separated extensions to include")
    ap.add_argument("--start", type=int, default=0, help="Start index after sorting/filtering")
    ap.add_argument("--max", type=int, default=0, help="Max files to view (0 = all)")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle selected files before viewing")
    ap.add_argument("--seed", type=int, default=123, help="Random seed used with --shuffle")
    ap.add_argument("--axes", action="store_true", help="Show coordinate axes in viewer")
    ap.add_argument(
        "--kpt-count",
        type=int,
        default=3,
        help="Number of keypoints appended at end of each PCD (0 disables keypoint split)",
    )
    ap.add_argument(
        "--kpt-radius",
        type=float,
        default=0.25,
        help="Keypoint sphere radius in meters",
    )
    ap.add_argument(
        "--no-warning-check",
        action="store_true",
        help="Disable warning-box inside/outside checks for keypoints.",
    )
    ap.add_argument(
        "--warning-keypoint-csv",
        type=str,
        default="",
        help="CSV with per-scene keypoint confidences/names (default: <root>/keypoint_confidence.csv).",
    )
    ap.add_argument(
        "--warning-profile-csv",
        type=str,
        default=str(WARNING_PROFILE_CSV),
        help="Profile CSV used to resolve recommended warning YAML per aircraft.",
    )
    ap.add_argument(
        "--warning-yaml-column",
        type=str,
        default=str(WARNING_YAML_COLUMN),
        help="Column name in --warning-profile-csv containing warning YAML path.",
    )
    ap.add_argument(
        "--warning-yaml-root",
        type=str,
        default=str(WARNING_YAML_ROOT),
        help="Fallback root for warning YAML lookup.",
    )
    ap.add_argument(
        "--warning-yaml-relpath",
        type=str,
        default=str(WARNING_YAML_RELPATH),
        help="Relative YAML path under each aircraft folder in --warning-yaml-root.",
    )
    ap.add_argument(
        "--warning-target-level",
        type=int,
        default=int(WARNING_TARGET_LEVEL),
        help="Preferred warning_level in YAML crop_boxes (fallbacks to all boxes if not found).",
    )
    ap.add_argument(
        "--warning-box-scale",
        type=float,
        default=1.0,
        help="Scale factor on warning-box half sizes for inside/outside checks.",
    )
    ap.add_argument(
        "--warning-h5-root",
        type=str,
        default=str(SOURCE),
        help="Root folder containing source H5 files (used for scene-wise warning-box transform).",
    )
    ap.add_argument(
        "--no-warning-scene-transform",
        action="store_true",
        help="Disable scene-wise warning-box transform from source H5 keypoints.",
    )
    ap.add_argument(
        "--warning-fallback-kp-names",
        type=str,
        default="front_wheels_mid,engine_left_box_center,engine_right_box_center",
        help=(
            "Comma-separated fallback keypoint names used when scene is missing in "
            "--warning-keypoint-csv (applied in tail keypoint order)."
        ),
    )
    ap.add_argument(
        "--warning-conf-threshold",
        type=float,
        default=-1.0,
        help=(
            "Optional minimum keypoint confidence used for keypoint PASS/FAIL confidence stats. "
            "Use negative value to disable threshold filtering."
        ),
    )
    ap.add_argument(
        "--min-warning-fail-kp-to-visualize",
        type=int,
        default=0,
        help="If >0, open windows only for scenes with at least this many failed warning keypoints.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if args.file:
        p = Path(args.file).expanduser().resolve()
        if not p.exists() or not p.is_file():
            raise RuntimeError(f"--file not found: {p}")
        files = [p]
        root_for_defaults = p.parent
    else:
        root_for_defaults = Path(args.root).expanduser().resolve()
        exts = [x.strip() for x in str(args.exts).split(",") if x.strip()]
        files = _collect_cloud_files(
            root=root_for_defaults,
            recursive=bool(args.recursive),
            pattern=str(args.pattern or ""),
            exts=exts,
        )

        if not files:
            raise RuntimeError("No point-cloud files matched.")

        start = max(0, int(args.start))
        if start >= len(files):
            raise RuntimeError(f"--start {start} is >= number of files ({len(files)}).")
        files = files[start:]

        max_n = int(args.max)
        if max_n > 0:
            files = files[:max_n]

        if args.shuffle:
            rng = random.Random(int(args.seed))
            rng.shuffle(files)

    if args.warning_keypoint_csv:
        kp_csv = Path(args.warning_keypoint_csv).expanduser().resolve()
    else:
        kp_csv = (root_for_defaults / "keypoint_confidence.csv").expanduser().resolve()

    print(f"[viewer] files selected: {len(files)}")
    print(f"[viewer] code-toggle VISUALIZATION_ENABLED={bool(VISUALIZATION_ENABLED)}")
    print(f"[viewer] code-toggle WARNING_PASS_FAIL_ENABLED={bool(WARNING_PASS_FAIL_ENABLED)}")
    fallback_kp_names = _parse_name_csv(str(args.warning_fallback_kp_names))
    conf_thr = None
    try:
        if float(args.warning_conf_threshold) >= 0.0:
            conf_thr = float(args.warning_conf_threshold)
    except Exception:
        conf_thr = None
    _view_files(
        files,
        show_axes=bool(args.axes),
        visualize=bool(VISUALIZATION_ENABLED),
        kpt_count=int(args.kpt_count),
        kpt_radius=float(args.kpt_radius),
        warning_check_enabled=not bool(args.no_warning_check),
        warning_pass_fail_enabled=bool(WARNING_PASS_FAIL_ENABLED),
        warning_keypoint_csv=kp_csv,
        warning_profile_csv=str(args.warning_profile_csv or ""),
        warning_yaml_column=str(args.warning_yaml_column or WARNING_YAML_COLUMN),
        warning_yaml_root=str(args.warning_yaml_root or WARNING_YAML_ROOT),
        warning_yaml_relpath=str(args.warning_yaml_relpath or WARNING_YAML_RELPATH),
        warning_target_level=int(args.warning_target_level),
        warning_box_scale=float(args.warning_box_scale),
        warning_fallback_kp_names=fallback_kp_names,
        warning_h5_root=str(args.warning_h5_root or SOURCE),
        use_scene_h5_transform=not bool(args.no_warning_scene_transform),
        warning_conf_threshold=conf_thr,
        min_warning_fail_kp_to_visualize=int(max(0, args.min_warning_fail_kp_to_visualize)),
    )


if __name__ == "__main__":
    main()
