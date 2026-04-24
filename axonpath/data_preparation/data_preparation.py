import os
import json
import hashlib
from skimage import io
from glob import glob
import random
import numpy as np
import csv
from skimage.measure import label
from skimage.segmentation import clear_border
from axonpath.utils.image_processing import fill_labels
from axonpath.skeleton.skeleton_aware_distance_transform import LabelDistanceTransforms

_MANIFEST_FILE = "manifest.json"


def _hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_manifest(out_root: str) -> dict:
    manifest_path = os.path.join(out_root, _MANIFEST_FILE)
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            return json.load(f)
    return {"sources": {}}


def _save_manifest(out_root: str, manifest: dict) -> None:
    with open(os.path.join(out_root, _MANIFEST_FILE), "w") as f:
        json.dump(manifest, f, indent=2)


def shuffle_tuples_in_list(list1, list2, list3):
    assert len(list1) == len(list2) == len(list3)
    list_of_tuples = list(zip(list1, list2, list3))
    random.shuffle(list_of_tuples)
    list1, list2, list3 = zip(*list_of_tuples)
    return list(list1), list(list2), list(list3)


def compute_tile_size(h, w, min_size=512, max_padding_ratio=0.1): 
    def compute_dim_tile_size(size, min_size, max_padding_ratio):
        pad = (min_size - size % min_size) % min_size
        if pad <= min_size * max_padding_ratio:
            return min_size, pad
        for ts in range(min_size + 1, 2000):
            pad = (ts - size % ts) % ts
            if pad <= ts * max_padding_ratio:
                return ts, pad
        return min_size, (min_size - size % min_size) % min_size

    tile_h, pad_h = compute_dim_tile_size(h, min_size, max_padding_ratio)
    tile_w, pad_w = compute_dim_tile_size(w, min_size, max_padding_ratio)
    return tile_h, tile_w, pad_h, pad_w


def tile_image(image, tile_h, tile_w):
    h, w = image.shape[:2]
    tiles = []
    for y in range(0, h, tile_h):
        for x in range(0, w, tile_w):
            tiles.append(image[y:y+tile_h, x:x+tile_w])
    return tiles


def apply_padding(im, pad_h, pad_w, fill_value=0):
    if im.ndim == 3:
        return np.pad(im, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=fill_value)
    else:
        return np.pad(im, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=fill_value)


def fix_label_edge_padding(fibre_lbl, axon_lbl, mask_sem):
    fixed = False
    h, w = fibre_lbl.shape
    new_fibre = fibre_lbl.copy()
    new_axon = axon_lbl.copy()
    new_mask = mask_sem.copy()

    if h >= 2:
        if np.any(fibre_lbl[-2, :] > 0) and np.all(fibre_lbl[-1, :] == 0):
            new_fibre[-1, :] = new_fibre[-2, :]
            new_axon[-1, :] = new_axon[-2, :]
            new_mask[-1, :] = new_mask[-2, :]
            fixed = True

    if w >= 2:
        if np.any(fibre_lbl[:, -2] > 0) and np.all(fibre_lbl[:, -1] == 0):
            new_fibre[:, -1] = new_fibre[:, -2]
            new_axon[:, -1] = new_axon[:, -2]
            new_mask[:, -1] = new_mask[:, -2]
            fixed = True

    return new_fibre, new_axon, new_mask, fixed


def split_tiles(tiles, create_test_split=True, seed=None):
    """
    Split tiles into train, val, (optional) test sets.

    Args:
        tiles (list): List of tile data.
        create_test_split (bool): Whether to create a separate test split.
        seed (int, optional): Random seed for reproducible splits.

    Returns:
        tuple: train_tiles, val_tiles, test_tiles

    Note: splitting is tile-level, not image-level. Tiles from the same source
    image can appear in different splits, which is a known limitation. Image-level
    splitting is preferred in principle but impractical when the dataset contains
    only a few large WSI images (e.g. 3 BF images), where image-level splitting
    would leave only 1 image in val — too few for reliable metrics.
    """
    if seed is not None:
        random.seed(seed)
    random.shuffle(tiles)
    n_total = len(tiles)

    if n_total < 1:
        raise ValueError("No tiles to split.")

    if n_total >= 10:
        n_train = round(n_total * 0.8)
        remaining = n_total - n_train
        n_val = max(round(n_total * 0.1), remaining // 2)
        n_test = remaining - n_val
    elif n_total >= 3:
        n_train = n_total - 2
        n_val = 1
        n_test = 1
    else:
        n_train = n_total - 1
        n_val = 1
        n_test = 0

    # Merge test into train if test split is disabled
    if not create_test_split:
        n_train += n_test
        n_test = 0

    return tiles[:n_train], tiles[n_train:n_train+n_val], tiles[n_train+n_val:]



def split_dataset(in_root, out_root, fix_label_padding=True, create_test_split=True, seed=None):
    os.makedirs(out_root, exist_ok=True)

    # Guard: abort if tiles exist without a manifest (pre-manifest prepared data)
    existing_tiles = glob(os.path.join(out_root, "*_tiles", "*.npy"))
    manifest_path = os.path.join(out_root, _MANIFEST_FILE)
    if existing_tiles and not os.path.exists(manifest_path):
        print(
            f"ERROR: {out_root} contains prepared tiles but no manifest. "
            "Re-running would create duplicate tiles. "
            "Use --overwrite to delete all existing output and reprocess from scratch."
        )
        return

    manifest = _load_manifest(out_root)
    summary_records = []
    tile_records = []

    for dataset in os.listdir(in_root):
        dataset_path = os.path.join(in_root, dataset)
        if not os.path.isdir(dataset_path):
            continue

        dataset_key = os.path.realpath(dataset_path)
        if dataset_key in manifest["sources"]:
            print(f"Skipping {dataset} (already in manifest - use --overwrite to reprocess).")
            continue

        image_paths = sorted(glob(os.path.join(dataset_path, 'images', '*.tif')))
        mask_paths = sorted(glob(os.path.join(dataset_path, 'masks', '*.tif')))
        label_paths = sorted(glob(os.path.join(dataset_path, 'labels', '*.tif')))
        if not image_paths or len(image_paths) != len(mask_paths) or len(mask_paths) != len(label_paths):
            print(f"Skipping {dataset_path}: inconsistent file counts.")
            continue

        print(f"Processing dataset: {dataset} ({len(image_paths)} images)")
        image_paths, mask_paths, label_paths = shuffle_tuples_in_list(image_paths, mask_paths, label_paths)

        all_tiles = []
        tile_counts = {}

        for img_path, msk_path, lbl_path in zip(image_paths, mask_paths, label_paths):
            img = io.imread(img_path).astype(np.float32)

            # Warn and convert accidental RGB inputs to grayscale
            if img.ndim == 3:
                print(
                    f"  WARNING: {os.path.basename(img_path)} loaded as RGB ({img.shape}). "
                    "Converting to grayscale (mean of channels). Check that this is intended."
                )
                img = img.mean(axis=2).astype(np.float32)

            mask = io.imread(msk_path).astype(np.uint8)
            axon_mask = (mask == 3).astype(np.uint8)
            label_raw = io.imread(lbl_path).astype(np.uint16)

            # Optional edge fix
            if fix_label_padding:
                label_raw, axon_mask, mask, _ = fix_label_edge_padding(label_raw, axon_mask, mask)

            # Fill fibre labels
            filled_label = fill_labels(label_raw.astype(np.int32))
            unique_values = np.unique(filled_label)
            num_fibers = len(unique_values[unique_values != 0])

            # Fill axon labels
            axon_instance = fill_labels(label(axon_mask.astype(np.int32)))

            # Compute SDTs
            sdt_fibre, _, _ = LabelDistanceTransforms(filled_label, 0.3, False, False, True).skeleton_aware_dist_trans()
            sdt_axon, _, _ = LabelDistanceTransforms(axon_instance, 0.15, False, False, True).skeleton_aware_dist_trans()

            h, w = label_raw.shape
            tile_h, tile_w, pad_h, pad_w = compute_tile_size(h, w)

            if h < 512 or w < 512:
                raise ValueError(
                    f"Source image '{os.path.basename(img_path)}' is {h}x{w}, which is smaller "
                    "than the minimum tile size (512x512). Resize or remove it from the dataset."
                )

            img_padded = apply_padding(img, pad_h, pad_w)
            mask_padded = apply_padding(mask, pad_h, pad_w)
            label_padded = apply_padding(filled_label, pad_h, pad_w)
            axon_instance_padded = apply_padding(axon_instance, pad_h, pad_w)
            sdt_fibre_padded = apply_padding(sdt_fibre, pad_h, pad_w, fill_value=-1)
            sdt_axon_padded = apply_padding(sdt_axon, pad_h, pad_w, fill_value=-1)

            img_tiles = tile_image(img_padded, tile_h, tile_w)
            mask_tiles = tile_image(mask_padded, tile_h, tile_w)
            label_tiles = tile_image(label_padded, tile_h, tile_w)
            sdt_fibre_tiles = tile_image(sdt_fibre_padded, tile_h, tile_w)
            sdt_axon_tiles = tile_image(sdt_axon_padded, tile_h, tile_w)
            axon_tiles = tile_image(axon_instance_padded, tile_h, tile_w)

            base_name = os.path.splitext(os.path.basename(img_path))[0]
            tile_count = len(img_tiles)
            tile_counts[base_name] = {'count': tile_count, 'num_fibers': num_fibers, 'h': h, 'w': w}

            valid_count = 0
            for i, (im_tile, msk_tile, lbl_tile, axon_tile, sdt_fibre_tile, sdt_axon_tile) in enumerate(
                zip(img_tiles, mask_tiles, label_tiles, axon_tiles, sdt_fibre_tiles, sdt_axon_tiles)
            ):
                if not np.any(clear_border(lbl_tile) > 0):
                    continue

                tile_name = f"{dataset}_{base_name}_tile{valid_count}"
                all_tiles.append((tile_name, im_tile, msk_tile, lbl_tile, axon_tile, sdt_fibre_tile, sdt_axon_tile, dataset, base_name))
                valid_count += 1

            tile_counts[base_name] = {
                'count': valid_count,
                'num_fibers': num_fibers,
                'h': h,
                'w': w
            }
            print(f"  Tiled {base_name}: {valid_count} valid tiles (skipped {tile_count - valid_count})")

        print(f"Splitting {len(all_tiles)} tiles...")
        train_tiles, val_tiles, test_tiles = split_tiles(all_tiles, create_test_split, seed=seed)

        for split_name, tiles in zip(['train', 'val', 'test'], [train_tiles, val_tiles, test_tiles]):
            for tile_name, im_tile, msk_tile, lbl_tile, axon_tile, sdt_fibre_tile, sdt_axon_tile, dataset, base_name in tiles:
                out_tile_path = os.path.join(out_root, f"{split_name}_tiles")
                os.makedirs(out_tile_path, exist_ok=True)

                tile_data = {
                    "image": im_tile,
                    "mask_sem": msk_tile,
                    "mask_fibre": lbl_tile,
                    "mask_axon": axon_tile,
                    "sdt_fibre": sdt_fibre_tile,
                    "sdt_axon": sdt_axon_tile,
                }
                np.save(os.path.join(out_tile_path, f"{tile_name}.npy"), tile_data)

                tile_records.append([split_name, tile_name, dataset, im_tile.shape[0], im_tile.shape[1]])
                tile_counts[base_name][f'{split_name}_count'] = tile_counts[base_name].get(f'{split_name}_count', 0) + 1

        for base_name, stats in tile_counts.items():
            summary_records.append([
                dataset,
                base_name,
                stats['h'],
                stats['w'],
                stats['num_fibers'],
                stats['count'],
                stats.get('train_count', 0),
                stats.get('val_count', 0),
                stats.get('test_count', 0),
            ])

        # Compute source file hashes and record in manifest
        all_source_files = image_paths + mask_paths + label_paths
        source_file_hashes = {}
        for fp in all_source_files:
            rel = os.path.relpath(fp, dataset_path)
            source_file_hashes[rel] = _hash_file(fp)

        total_saved_tiles = sum(
            tile_counts[bn].get("train_count", 0) +
            tile_counts[bn].get("val_count", 0) +
            tile_counts[bn].get("test_count", 0)
            for bn in tile_counts
        )
        manifest["sources"][dataset_key] = {
            "tile_count": total_saved_tiles,
            "source_file_hashes": source_file_hashes,
        }
        _save_manifest(out_root, manifest)
        print(f"Finished dataset: {dataset}\n")

    summary_path = os.path.join(out_root, "dataset_summary.tsv")
    with open(summary_path, mode='w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['Dataset', 'ImageName', 'Height', 'Width', 'NumFibers', 'NumTiles', 'TrainSplit', 'ValSplit', 'TestSplit'])
        writer.writerows(summary_records)

    tile_list_path = os.path.join(out_root, "tiles_info.tsv")
    with open(tile_list_path, mode='w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['Split', 'ImageName', 'Dataset', 'Height', 'Width'])
        writer.writerows(tile_records)

    print(f"\n Done! Saved:\n  - Summary: {summary_path}\n  - Tile list: {tile_list_path}\n  - Total tiles: {len(tile_records)} from {len(summary_records)} images.\n")