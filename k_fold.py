#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Set


# =========================
# CONFIG (NO ARGS)
# =========================
DATA_ROOT = Path("aircraft_pose_all4_kfold")   # dataset that contains images/labels
BAG_SPLIT_CSV = DATA_ROOT / "bag_split.csv"   # must exist here (or change to original dataset path)
K = 5
SEED = 123
OUT_DIR = DATA_ROOT / "kfold"
INCLUDE_ORIGINAL_TEST = False
MAKE_SYMLINKS = True


# =========================
# CSV reader (skips # lines)
# =========================
def read_bag_split(csv_path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with csv_path.open("r", newline="") as f:
        # skip comments
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                raise RuntimeError("CSV has no header.")
            s = line.strip()
            if s and not s.startswith("#"):
                f.seek(pos)
                break

        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise RuntimeError("Could not read CSV header.")
        for row in reader:
            if not row:
                continue
            rows.append({k: (v.strip() if isinstance(v, str) else v) for k, v in row.items()})
    return rows


@dataclass
class BagInfo:
    bag_stem: str
    n_scenes: int
    split: str


def build_bags(rows: List[Dict[str, str]]) -> List[BagInfo]:
    out: List[BagInfo] = []
    seen = set()
    for r in rows:
        stem = r["bag_stem"]
        if stem in seen:
            continue
        seen.add(stem)
        split = (r.get("split") or "").lower()
        try:
            n_scenes = int(r.get("n_scenes", "0"))
        except Exception:
            n_scenes = 0
        out.append(BagInfo(bag_stem=stem, n_scenes=n_scenes, split=split))
    return out


# =========================
# file ops
# =========================
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, make_symlinks: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if make_symlinks:
        os.symlink(src.resolve(), dst)
    else:
        shutil.copy2(src, dst)


def collect_pairs(data_root: Path, splits: List[str]) -> Tuple[Dict[str, Path], Dict[str, Path]]:
    img_by_stem: Dict[str, Path] = {}
    lbl_by_stem: Dict[str, Path] = {}

    for sp in splits:
        img_dir = data_root / "images" / sp
        lbl_dir = data_root / "labels" / sp
        if not img_dir.is_dir() or not lbl_dir.is_dir():
            continue
        for p in img_dir.glob("*.png"):
            img_by_stem[p.stem] = p
        for p in lbl_dir.glob("*.txt"):
            lbl_by_stem[p.stem] = p

    return img_by_stem, lbl_by_stem


def read_original_yaml(yaml_path: Path) -> str:
    return yaml_path.read_text()


def write_fold_yaml(out_yaml: Path, fold_root: Path, orig_yaml_text: str):
    lines = orig_yaml_text.splitlines()
    new_lines = []
    for ln in lines:
        if ln.strip().startswith("path:"):
            new_lines.append(f"path: {fold_root.resolve()}")
        elif ln.strip().startswith("train:"):
            new_lines.append("train: images/train")
        elif ln.strip().startswith("val:"):
            new_lines.append("val: images/val")
        elif ln.strip().startswith("test:"):
            new_lines.append("test: images/test")
        else:
            new_lines.append(ln)
    out_yaml.write_text("\n".join(new_lines) + "\n")


# =========================
# robust mapping: scene_stem -> bag_stem (FIXED)
# =========================
def scene_to_bag(stem: str) -> str | None:
    """
    Exported scene stems look like:
      <bag_stem>__<scene_name>

    But bag_stem itself contains "__" (e.g. movement_a380_800__2025-08-28T05-43-59),
    so we must take everything BEFORE the LAST "__".
    """
    parts = stem.split("__")
    if len(parts) < 2:
        return None
    return "__".join(parts[:-1])


def build_bag_to_scenes(scene_stems: List[str], valid_bag_stems: Set[str]) -> Dict[str, List[str]]:
    bag_to_scenes: Dict[str, List[str]] = {b: [] for b in valid_bag_stems}
    unknown = 0
    for s in scene_stems:
        b = scene_to_bag(s)
        if b is None or b not in valid_bag_stems:
            unknown += 1
            continue
        bag_to_scenes[b].append(s)

    # remove empty bags
    bag_to_scenes = {b: ss for b, ss in bag_to_scenes.items() if ss}
    print(f"[kfold] Scene stems total={len(scene_stems)}, matched_bags={len(bag_to_scenes)}, unmatched_scenes={unknown}")
    return bag_to_scenes


# =========================
# K folds with balance by scene count (actual exported scene count)
# =========================
def make_k_folds(bags: List[str], bag_to_scenes: Dict[str, List[str]], k: int, seed: int) -> List[List[str]]:
    rng = random.Random(seed)
    bags_shuf = bags[:]
    rng.shuffle(bags_shuf)

    # sort by actual exported scenes per bag, largest first
    bags_sorted = sorted(bags_shuf, key=lambda b: len(bag_to_scenes[b]), reverse=True)

    folds: List[List[str]] = [[] for _ in range(k)]
    counts = [0] * k
    for b in bags_sorted:
        idx = min(range(k), key=lambda i: counts[i])
        folds[idx].append(b)
        counts[idx] += len(bag_to_scenes[b])
    return folds


def main():
    print("[kfold] DATA_ROOT =", DATA_ROOT.resolve())
    print("[kfold] Reading bag_split.csv:", BAG_SPLIT_CSV)

    if not BAG_SPLIT_CSV.exists():
        raise FileNotFoundError(f"bag_split.csv not found: {BAG_SPLIT_CSV.resolve()}")

    rows = read_bag_split(BAG_SPLIT_CSV)
    bags_all = build_bags(rows)

    test_bags = [b for b in bags_all if b.split == "test"]
    cv_bags = [b for b in bags_all if b.split != "test"]

    print(f"[kfold] Bags total: {len(bags_all)} | CV bags: {len(cv_bags)} | test bags: {len(test_bags)}")

    # index exported files on disk (train+val only for CV)
    img_by_stem, lbl_by_stem = collect_pairs(DATA_ROOT, ["train", "val"])
    common_stems = sorted(set(img_by_stem.keys()) & set(lbl_by_stem.keys()))
    if not common_stems:
        raise RuntimeError("No matching image/label stems found under images/train+val and labels/train+val.")

    # map bag -> scenes using stems on disk
    cv_bag_stems = {b.bag_stem for b in cv_bags}
    bag_to_scenes = build_bag_to_scenes(common_stems, cv_bag_stems)

    # IMPORTANT: only use bags that actually exist on disk
    cv_bags_present = [b for b in cv_bags if b.bag_stem in bag_to_scenes]
    missing_bags = [b.bag_stem for b in cv_bags if b.bag_stem not in bag_to_scenes]
    print(f"[kfold] CV bags present on disk: {len(cv_bags_present)} | missing: {len(missing_bags)}")
    if missing_bags:
        print("[kfold] Example missing bag stems:", missing_bags[:5])

    if len(cv_bags_present) < K:
        raise RuntimeError(f"Not enough CV bags on disk ({len(cv_bags_present)}) for K={K}")

    cv_bag_names = [b.bag_stem for b in cv_bags_present]
    folds_val_bags = make_k_folds(cv_bag_names, bag_to_scenes, K, SEED)

    orig_yaml = DATA_ROOT / "aircraft_pose.yaml"
    if not orig_yaml.exists():
        raise FileNotFoundError(f"Original YAML not found: {orig_yaml.resolve()}")
    orig_yaml_text = read_original_yaml(orig_yaml)

    # prepare output
    ensure_dir(OUT_DIR)

    for fi in range(K):
        val_bags = set(folds_val_bags[fi])
        train_bags = set(b for j in range(K) if j != fi for b in folds_val_bags[j])

        val_stems = [s for b in val_bags for s in bag_to_scenes[b]]
        train_stems = [s for b in train_bags for s in bag_to_scenes[b]]

        fold_root = OUT_DIR / f"fold_{fi}"
        if fold_root.exists():
            shutil.rmtree(fold_root)

        # dirs
        ensure_dir(fold_root / "images" / "train")
        ensure_dir(fold_root / "images" / "val")
        ensure_dir(fold_root / "labels" / "train")
        ensure_dir(fold_root / "labels" / "val")

        # link/copy
        for s in train_stems:
            link_or_copy(img_by_stem[s], fold_root / "images" / "train" / f"{s}.png", MAKE_SYMLINKS)
            link_or_copy(lbl_by_stem[s], fold_root / "labels" / "train" / f"{s}.txt", MAKE_SYMLINKS)

        for s in val_stems:
            link_or_copy(img_by_stem[s], fold_root / "images" / "val" / f"{s}.png", MAKE_SYMLINKS)
            link_or_copy(lbl_by_stem[s], fold_root / "labels" / "val" / f"{s}.txt", MAKE_SYMLINKS)

        # optional test passthrough
        if INCLUDE_ORIGINAL_TEST:
            img_test, lbl_test = collect_pairs(DATA_ROOT, ["test"])
            test_common = sorted(set(img_test.keys()) & set(lbl_test.keys()))
            ensure_dir(fold_root / "images" / "test")
            ensure_dir(fold_root / "labels" / "test")
            for s in test_common:
                link_or_copy(img_test[s], fold_root / "images" / "test" / f"{s}.png", MAKE_SYMLINKS)
                link_or_copy(lbl_test[s], fold_root / "labels" / "test" / f"{s}.txt", MAKE_SYMLINKS)

        # yaml
        out_yaml = fold_root / f"aircraft_pose_fold_{fi}.yaml"
        write_fold_yaml(out_yaml, fold_root, orig_yaml_text)

        print(f"[kfold] fold {fi}: train_bags={len(train_bags)} val_bags={len(val_bags)} "
              f"| train_files={len(train_stems)} val_files={len(val_stems)}")
        print(f"[kfold] Wrote YAML: {out_yaml}")

    print("\n[kfold] Done.")
    print(f"[kfold] Train fold 0 example:\n"
          f"  yolo pose train model=yolov8s-pose.pt data={OUT_DIR / 'fold_0' / 'aircraft_pose_fold_0.yaml'} imgsz=1024 epochs=150 ...")


if __name__ == "__main__":
    main()
