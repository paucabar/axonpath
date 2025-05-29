import os 
import argparse
from skimage import io
from glob import glob
import random
import numpy as np
import csv
from image_processing import fill_labels

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
        pad = (min_size - size % min_size) % min_size
        return min_size, pad

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

def split_tiles(tiles):
    random.shuffle(tiles)
    n_total = len(tiles)
    if n_total >= 10:
        n_train = round(n_total * 0.8)
        n_val = round(n_total * 0.1)
    elif n_total >= 3:
        n_train = 1
        n_val = 1
    else:
        n_train = 1
        n_val = 0
    n_test = n_total - n_train - n_val
    return tiles[:n_train], tiles[n_train:n_train+n_val], tiles[n_train+n_val:]

def create_train_val_test_split_all(in_root, out_root):
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
            label_raw = io.imread(lbl_path)

            filled_label = fill_labels(label_raw.astype(np.int32))
            unique_values = np.unique(filled_label)
            num_fibers = len(unique_values[unique_values != 0])

            h, w = label_raw.shape
            tile_h, tile_w, pad_h, pad_w = compute_tile_size(h, w)

            img_padded = apply_padding(img, pad_h, pad_w)
            mask_padded = apply_padding(mask, pad_h, pad_w)
            label_padded = apply_padding(filled_label, pad_h, pad_w)

            img_tiles = tile_image(img_padded, tile_h, tile_w)
            mask_tiles = tile_image(mask_padded, tile_h, tile_w)
            label_tiles = tile_image(label_padded, tile_h, tile_w)

            base_name = os.path.splitext(os.path.basename(img_path))[0]
            tile_count = len(img_tiles)
            tile_counts[base_name] = {'count': tile_count, 'num_fibers': num_fibers, 'h': h, 'w': w}

            for i, (im_tile, msk_tile, lbl_tile) in enumerate(zip(img_tiles, mask_tiles, label_tiles)):
                tile_name = f"{base_name}_tile{i}.tif"
                all_tiles.append((tile_name, im_tile, msk_tile, lbl_tile, dataset, base_name))

            print(f"  Tiled {base_name}: {tile_count} tiles")

        print(f"Splitting {len(all_tiles)} tiles...")
        train_tiles, val_tiles, test_tiles = split_tiles(all_tiles)

        for split_name, tiles in zip(['train', 'val', 'test'], [train_tiles, val_tiles, test_tiles]):
            for tile_name, im_tile, msk_tile, lbl_tile, dataset, base_name in tiles:
                out_img_path = os.path.join(out_root, f"{split_name}_images")
                out_mask_path = os.path.join(out_root, f"{split_name}_masks")
                out_label_path = os.path.join(out_root, f"{split_name}_labels")
                os.makedirs(out_img_path, exist_ok=True)
                os.makedirs(out_mask_path, exist_ok=True)
                os.makedirs(out_label_path, exist_ok=True)

                io.imsave(os.path.join(out_img_path, tile_name), im_tile, check_contrast=False)
                io.imsave(os.path.join(out_mask_path, tile_name), msk_tile, check_contrast=False)
                io.imsave(os.path.join(out_label_path, tile_name), lbl_tile.astype(np.uint16), check_contrast=False)

                tile_records.append([split_name, tile_name, dataset, im_tile.shape[0], im_tile.shape[1]])

                if base_name in tile_counts:
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
    parser.add_argument('--input', type=str, default="datasets", help='Input root folder with datasets')
    parser.add_argument('--output', type=str, default="prepared_data", help='Output root folder')
    args = parser.parse_args()

    create_train_val_test_split_all(args.input, args.output)

if __name__ == "__main__":
    main()
