import os
import argparse
from skimage import io
from glob import glob
import random
import numpy as np
import csv
from skimage.measure import label
from aimsegdl.utils.image_processing import fill_labels
from aimsegdl.skeleton.skeleton_aware_distance_transform import LabelDistanceTransforms


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


def apply_padding(im, pad_h, pad_w):
    if im.ndim == 3:
        return np.pad(im, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
    else:
        return np.pad(im, ((0, pad_h), (0, pad_w)), mode='constant')


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


def split_tiles(tiles):
    random.shuffle(tiles)
    n_total = len(tiles)
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
        n_train = 1
        n_val = 0
        n_test = n_total - 1
    return tiles[:n_train], tiles[n_train:n_train+n_val], tiles[n_train+n_val:]


def create_train_val_test_split_all(in_root, out_root, fix_label_padding=True):
    os.makedirs(out_root, exist_ok=True)
    summary_records = []
    tile_records = []

    for dataset in os.listdir(in_root):
        dataset_path = os.path.join(in_root, dataset)
        if not os.path.isdir(dataset_path):
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
            img = io.imread(img_path)
            mask = io.imread(msk_path)
            axon_mask = (mask == 3).astype(np.uint8)
            label_raw = io.imread(lbl_path)

            # Optional edge fix
            if fix_label_padding:
                label_raw, axon_mask, mask, _ = fix_label_edge_padding(label_raw, axon_mask, mask)

            # Fill fibre labels
            filled_label = fill_labels(label_raw.astype(np.int16))
            unique_values = np.unique(filled_label)
            num_fibers = len(unique_values[unique_values != 0])

            # Fill axon labels
            axon_instance = fill_labels(label(axon_mask.astype(np.int16)))

            # Compute SDTs
            sdt_fibre, _, _ = LabelDistanceTransforms(filled_label, 0.3, False, False, True).skeleton_aware_dist_trans()
            sdt_axon, _, _ = LabelDistanceTransforms(axon_instance, 0.15, False, False, True).skeleton_aware_dist_trans()

            h, w = label_raw.shape
            tile_h, tile_w, pad_h, pad_w = compute_tile_size(h, w)

            img_padded = apply_padding(img, pad_h, pad_w)
            mask_padded = apply_padding(mask, pad_h, pad_w)
            label_padded = apply_padding(filled_label, pad_h, pad_w)
            axon_instance_padded = apply_padding(axon_instance, pad_h, pad_w)
            sdt_fibre_padded = apply_padding(sdt_fibre, pad_h, pad_w)
            sdt_axon_padded = apply_padding(sdt_axon, pad_h, pad_w)

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
                if not np.any(lbl_tile > 0):
                    continue

                tile_name = f"{base_name}_tile{valid_count}"
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
        train_tiles, val_tiles, test_tiles = split_tiles(all_tiles)

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


def main():
    parser = argparse.ArgumentParser(description="Prepare U-Net training data from datasets.")
    parser.add_argument('--input', type=str, default="../datasets", help='Input root folder with datasets')
    parser.add_argument('--output', type=str, default="../prepared_data", help='Output root folder')
    parser.add_argument('--fix_label_padding', action='store_true', default=True, help='Fix extra padded label rows/cols')
    args = parser.parse_args()

    create_train_val_test_split_all(args.input, args.output, fix_label_padding=args.fix_label_padding)


if __name__ == "__main__":
    main()