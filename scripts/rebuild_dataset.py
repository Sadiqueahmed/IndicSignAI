"""
rebuild_dataset.py — Re-extract landmark features with correct normalisation
==============================================================================
This script walks a labelled image dataset directory, extracts MediaPipe
hand landmarks with bounding-box normalisation (identical to inference),
and outputs:
  1.  keypoint.csv              — feature CSV for training
  2.  models/class_mapping.json — index-to-label mapping for fast_predict()

Usage:
    python scripts/rebuild_dataset.py --dataset path/to/images
    python scripts/rebuild_dataset.py --dataset training/data/images --output training/data/keypoint.csv

Directory layout expected:
    <dataset>/
        <CLASS_NAME_1>/
            img001.jpg
            img002.jpg
        <CLASS_NAME_2>/
            ...
"""

import os
import sys
import json
import argparse

import cv2
import numpy as np
import mediapipe as mp

# ── Project root for imports ──────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _REPO_ROOT)

from src.utils.landmark_utils import extract_and_normalize_landmarks, LANDMARK_FEATURE_DIM


def build_dataset(dataset_dir, output_csv, mp_confidence=0.80):
    """Scan labelled image folders and write feature CSV + class_mapping."""
    class_dirs = sorted([
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ])

    if not class_dirs:
        print(f"[ERROR] No class subdirectories found in {dataset_dir}")
        return

    print(f"Building dataset from {len(class_dirs)} classes in {dataset_dir}")
    print(f"  Output CSV:       {output_csv}")
    print(f"  Feature dim:      {LANDMARK_FEATURE_DIM}")
    print(f"  MP confidence:    {mp_confidence}")
    print()

    rows = []
    label_to_idx = {}
    idx_to_label = {}

    for cls_idx, class_name in enumerate(class_dirs):
        label_to_idx[class_name] = cls_idx
        idx_to_label[str(cls_idx)] = class_name

        class_path = os.path.join(dataset_dir, class_name)
        images = [f for f in os.listdir(class_path)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

        accepted = 0
        for img_file in images:
            img_path = os.path.join(class_path, img_file)
            frame = cv2.imread(img_path)
            if frame is None:
                continue

            landmark_vec, _ = extract_and_normalize_landmarks(
                frame, mp_confidence_gate=mp_confidence
            )
            if landmark_vec is None:
                continue

            row = [class_name] + landmark_vec.tolist()
            rows.append(row)
            accepted += 1

        print(f"  {class_name:30s}: {accepted:4d}/{len(images)} frames accepted")

    if not rows:
        print("\n[ERROR] No landmarks extracted. Check dataset or lower --confidence.")
        return

    # Write CSV
    import csv
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)
    print(f"\n[OK] Saved {len(rows)} rows → {output_csv}")

    # Write class_mapping.json
    mapping_path = os.path.join(_REPO_ROOT, 'models', 'class_mapping.json')
    os.makedirs(os.path.dirname(mapping_path), exist_ok=True)
    with open(mapping_path, 'w') as f:
        json.dump(idx_to_label, f, indent=2)
    print(f"[OK] Saved {len(idx_to_label)} classes → {mapping_path}")

    # Write label maps for reference
    data_dir = os.path.dirname(output_csv)
    with open(os.path.join(data_dir, 'label_to_id.json'), 'w') as f:
        json.dump(label_to_idx, f, indent=2)
    with open(os.path.join(data_dir, 'id_to_label.json'), 'w') as f:
        json.dump(idx_to_label, f, indent=2)
    print(f"[OK] Saved label maps → {data_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rebuild IndicSignAI training dataset')
    parser.add_argument('--dataset', required=True,
                        help='Path to labelled image dataset directory')
    parser.add_argument('--output', default='training/data/keypoint.csv',
                        help='Output CSV path (default: training/data/keypoint.csv)')
    parser.add_argument('--confidence', type=float, default=0.80,
                        help='MediaPipe detection confidence gate (default: 0.80)')
    args = parser.parse_args()

    build_dataset(args.dataset, args.output, args.confidence)
