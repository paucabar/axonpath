"""
Standalone annotation QC script.

Reads raw dataset directories (images / masks / labels triplets), runs the
annotation QC checks, and reports any issues — without touching prepared tile
directories. Safe to run while training is active.

Usage:
    python scripts/check_annotations.py \
        --input datasets_em datasets_bf \
        --output annotation_qc_check.tsv
"""

import argparse
import csv
import os
from glob import glob

import numpy as np
from skimage import io
from skimage.measure import label

from axonpath.data_preparation.data_preparation import _annotation_qc, fix_label_edge_padding
from axonpath.utils.image_processing import fill_labels


def check_dataset(dataset_path: str) -> list[tuple]:
    """Run QC on every image in a single dataset directory.

    Returns a list of (dataset_name, image_name, issue_type, count, detail) tuples.
    """
    dataset_name = os.path.basename(dataset_path)
    image_paths = sorted(glob(os.path.join(dataset_path, "images", "*.tif")))
    mask_paths  = sorted(glob(os.path.join(dataset_path, "masks",  "*.tif")))
    label_paths = sorted(glob(os.path.join(dataset_path, "labels", "*.tif")))

    if not image_paths:
        print(f"  Skipping {dataset_name}: no images found.")
        return []
    if len(image_paths) != len(mask_paths) or len(mask_paths) != len(label_paths):
        print(f"  Skipping {dataset_name}: mismatched file counts "
              f"(images={len(image_paths)}, masks={len(mask_paths)}, labels={len(label_paths)}).")
        return []

    records = []
    for img_path, msk_path, lbl_path in zip(image_paths, mask_paths, label_paths):
        base_name = os.path.splitext(os.path.basename(img_path))[0]

        mask      = io.imread(msk_path).astype(np.uint8)
        axon_mask = (mask == 3).astype(np.uint8)
        label_raw = io.imread(lbl_path).astype(np.uint16)

        label_raw, axon_mask, mask, _ = fix_label_edge_padding(label_raw, axon_mask, mask)
        filled_label = fill_labels(label_raw.astype(np.int32))

        issues = _annotation_qc(filled_label, mask)
        if issues:
            for issue_type, count, detail in issues:
                print(f"  QC WARNING [{base_name}] {issue_type}: {detail}")
                records.append((dataset_name, base_name, issue_type, count, detail))
        else:
            print(f"  OK [{base_name}]")

    return records


def main():
    parser = argparse.ArgumentParser(description="Standalone annotation QC check")
    parser.add_argument(
        "--input", nargs="+", required=True,
        help="One or more root directories containing dataset subdirectories "
             "(e.g. datasets_em datasets_bf)"
    )
    parser.add_argument(
        "--output", default="annotation_qc_check.tsv",
        help="Path for the output TSV report (default: annotation_qc_check.tsv)"
    )
    args = parser.parse_args()

    all_records = []

    for root_dir in args.input:
        if not os.path.isdir(root_dir):
            print(f"WARNING: {root_dir} is not a directory — skipping.")
            continue

        dataset_dirs = sorted(
            d for d in (os.path.join(root_dir, e) for e in os.listdir(root_dir))
            if os.path.isdir(d)
        )
        if not dataset_dirs:
            print(f"WARNING: no subdirectories found in {root_dir}.")
            continue

        for dataset_path in dataset_dirs:
            print(f"\nChecking {os.path.basename(dataset_path)} ...")
            records = check_dataset(dataset_path)
            all_records.extend(records)

    # Write TSV report
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["Dataset", "ImageName", "IssueType", "Count", "Detail"])
        writer.writerows(all_records)

    # Summary
    n_issues = len(all_records)
    affected_images = len({(r[0], r[1]) for r in all_records})
    print(f"\n=== QC summary ===")
    print(f"  Issues found : {n_issues}")
    print(f"  Images affected: {affected_images}")
    print(f"  Report written : {args.output}")


if __name__ == "__main__":
    main()
