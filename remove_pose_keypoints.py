#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Remove keypoints from an existing YOLO-pose dataset in-place.

What this updates:
1) All label .txt files under labels/ (removes triplets x y v by index)
2) aircraft_pose.yaml (updates keypoints list + kpt_shape)

Default mode is preview only. Use --apply to write changes.
"""

from __future__ import annotations

import argparse
import difflib
from pathlib import Path
from typing import List, Tuple

DEFAULT_DATASET_ROOT = Path(
    "/home/femi/yolo_pose_dataset_creation/aircraft_pose_with_normalising_applied_grayscale"
)


def _load_yaml_lines(yaml_path: Path) -> List[str]:
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML not found: {yaml_path}")
    return yaml_path.read_text(encoding="utf-8", errors="ignore").splitlines()


def _extract_keypoints_from_yaml(lines: List[str]) -> Tuple[List[str], int, int, str]:
    keypoints_line_idx = -1
    for i, ln in enumerate(lines):
        if ln.strip() == "keypoints:":
            keypoints_line_idx = i
            break
    if keypoints_line_idx < 0:
        raise ValueError("Could not find 'keypoints:' block in YAML.")

    names: List[str] = []
    start = keypoints_line_idx + 1
    i = start
    bullet_prefix = "  - "
    while i < len(lines):
        s = lines[i].lstrip()
        if not s.startswith("- "):
            break
        prefix_len = len(lines[i]) - len(lines[i].lstrip())
        bullet_prefix = (" " * prefix_len) + "- "
        names.append(s[2:].strip())
        i += 1

    if not names:
        raise ValueError("YAML has 'keypoints:' but contains no list entries.")
    return names, keypoints_line_idx, i, bullet_prefix


def _update_yaml_text(
    lines: List[str],
    keep_names: List[str],
    keypoints_start: int,
    keypoints_end: int,
    bullet_prefix: str,
) -> str:
    out = lines[:]

    # Replace kpt_shape line.
    for i, ln in enumerate(out):
        if ln.strip().startswith("kpt_shape:"):
            indent = ln[: len(ln) - len(ln.lstrip())]
            out[i] = f"{indent}kpt_shape: [{len(keep_names)}, 3]"
            break

    new_kp_lines = [f"{bullet_prefix}{nm}" for nm in keep_names]
    out = out[: keypoints_start + 1] + new_kp_lines + out[keypoints_end:]
    return "\n".join(out) + "\n"


def _iter_label_files(labels_root: Path) -> List[Path]:
    if not labels_root.exists() or not labels_root.is_dir():
        raise FileNotFoundError(f"Labels directory not found: {labels_root}")
    return sorted([p for p in labels_root.rglob("*.txt") if p.is_file()])


def _drop_triplets_from_line(line: str, drop_idx: set[int], expected_kps: int) -> Tuple[str, bool]:
    s = line.strip()
    if not s:
        return line, False
    toks = s.split()
    if len(toks) < 5:
        raise ValueError("Label row has fewer than 5 tokens.")
    rem = toks[5:]
    if len(rem) % 3 != 0:
        raise ValueError("Label row keypoint payload is not divisible by 3.")
    n_kps = len(rem) // 3
    if n_kps != expected_kps:
        raise ValueError(f"Label row has {n_kps} keypoints, expected {expected_kps}.")

    kept: List[str] = []
    for i in range(n_kps):
        if i in drop_idx:
            continue
        kept.extend(rem[3 * i : 3 * i + 3])

    new_line = " ".join(toks[:5] + kept)
    return new_line, (new_line != s)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Remove selected keypoints from YOLO-pose labels and aircraft_pose.yaml."
    )
    ap.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Dataset root containing labels/ and aircraft_pose.yaml",
    )
    ap.add_argument(
        "--yaml",
        type=Path,
        default=None,
        help="Path to aircraft_pose.yaml (default: <dataset-root>/aircraft_pose.yaml)",
    )
    ap.add_argument(
        "--labels-root",
        type=Path,
        default=None,
        help="Path to labels directory (default: <dataset-root>/labels)",
    )
    ap.add_argument(
        "--remove",
        nargs="+",
        default=None,
        help="Keypoint names to remove (exact names from YAML keypoints block).",
    )
    ap.add_argument(
        "--list",
        action="store_true",
        help="List keypoints with indices and exit.",
    )
    ap.add_argument(
        "--apply",
        action="store_true",
        help="Write changes. Without this flag, only preview is shown.",
    )
    args = ap.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    yaml_path = (args.yaml if args.yaml is not None else dataset_root / "aircraft_pose.yaml").expanduser().resolve()
    labels_root = (
        args.labels_root if args.labels_root is not None else dataset_root / "labels"
    ).expanduser().resolve()

    yaml_lines = _load_yaml_lines(yaml_path)
    kp_names, kp_start, kp_end, bullet_prefix = _extract_keypoints_from_yaml(yaml_lines)

    print(f"[dataset] {dataset_root}")
    print(f"[yaml]    {yaml_path}")
    print(f"[labels]  {labels_root}")

    if args.list or not args.remove:
        print("\n[keypoints]")
        for i, nm in enumerate(kp_names):
            print(f"  {i:2d}: {nm}")
        if not args.remove:
            print("\nProvide --remove <name1> [name2 ...] to remove keypoints.")
        return

    remove_names = [x.strip() for x in (args.remove or []) if x.strip()]
    if not remove_names:
        raise ValueError("No valid keypoint names provided to --remove.")

    missing = [nm for nm in remove_names if nm not in kp_names]
    if missing:
        print("[error] Unknown keypoint name(s):")
        for nm in missing:
            guess = difflib.get_close_matches(nm, kp_names, n=3, cutoff=0.4)
            if guess:
                print(f"  - {nm} (did you mean: {', '.join(guess)})")
            else:
                print(f"  - {nm}")
        raise SystemExit(2)

    drop_idx = {kp_names.index(nm) for nm in remove_names}
    keep_names = [nm for i, nm in enumerate(kp_names) if i not in drop_idx]
    if not keep_names:
        raise ValueError("Refusing to remove all keypoints; at least one must remain.")

    print("\n[plan]")
    print(f"  remove: {', '.join(remove_names)}")
    print(f"  old kpt count: {len(kp_names)}")
    print(f"  new kpt count: {len(keep_names)}")
    mode = "APPLY" if args.apply else "PREVIEW"
    print(f"  mode: {mode}")

    label_files = _iter_label_files(labels_root)
    if not label_files:
        print("[warn] No label files found.")
        return

    files_changed = 0
    rows_changed = 0
    files_error = 0

    for lp in label_files:
        txt = lp.read_text(encoding="utf-8", errors="ignore")
        lines = txt.splitlines()
        if not lines:
            continue

        changed_this_file = False
        new_lines: List[str] = []

        try:
            for ln in lines:
                if not ln.strip():
                    new_lines.append(ln)
                    continue
                new_ln, changed = _drop_triplets_from_line(
                    line=ln,
                    drop_idx=drop_idx,
                    expected_kps=len(kp_names),
                )
                new_lines.append(new_ln)
                if changed:
                    rows_changed += 1
                    changed_this_file = True
        except Exception as e:
            files_error += 1
            print(f"[warn] Skipped due to parse mismatch: {lp} ({e})")
            continue

        if changed_this_file:
            files_changed += 1
            if args.apply:
                lp.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    print("\n[labels]")
    print(f"  scanned files: {len(label_files)}")
    print(f"  changed files: {files_changed}")
    print(f"  changed rows:  {rows_changed}")
    print(f"  parse errors:  {files_error}")

    if files_changed == 0:
        print("[warn] No label rows changed. Nothing else to do.")
        return

    new_yaml_text = _update_yaml_text(
        lines=yaml_lines,
        keep_names=keep_names,
        keypoints_start=kp_start,
        keypoints_end=kp_end,
        bullet_prefix=bullet_prefix,
    )
    if args.apply:
        yaml_path.write_text(new_yaml_text, encoding="utf-8")
        print(f"[yaml] Updated: {yaml_path}")
        print("[done] Keypoint removal applied.")
    else:
        print("[preview] YAML would be updated (kpt_shape + keypoints list).")
        print("[preview] Re-run with --apply to write changes.")


if __name__ == "__main__":
    main()
