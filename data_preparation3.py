import os
import argparse
from skimage import io
from glob import glob
import random
import numpy as np
import pandas as pd
from image_processing import fill_labels

def compute_tile_size(h, w, min_size=512, max_padding=4):
    pad_h = (min_size - h % min_size) % min_size
    pad_w = (min_size - w % min_size) % min_size
    if pad_h <= max_padding and pad_w <= max_padding:
        return min_size, pad_h, pad_w
    for tile_size in range(min_size + 1, 2000):
        if h % tile_size == 0 and w % tile_size == 0:
            return tile_size, 0, 0
    return max(h, w), 0, 0

def tile_image(image, tile_size):
    h, w = image.shape[:2]
    tiles = []
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tiles.append((image[y:y+tile_size, x:x+tile_size], y, x))
    return tiles

def apply_padding(im, pad_h, pad_w):
    if im.ndim == 3:
        return np.pad(im, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
    else:
        return np.pad(im, ((0, pad_h), (0, pad_w)), mode='constant')

def create_train_val_test_split_all(in_root, out_root):
    os.makedirs(out_root, exist_ok=True)
    tile_records = []
    summary_records = []
    all_tiles = []

    print("Scanning datasets and processing images...")

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

        for img_path, msk_path, lbl_path in zip(image_paths, mask_paths, label_paths):
            print(f"Processing image: {img_path}")
            img = io.imread(img_path)
            mask = io.imread(msk_path)
            label_raw = io.imread(lbl_path)

            filled_label = fill_labels(label_raw.astype(np.int32))
            unique_values = np.unique(filled_label)
            num_fibers = len(unique_values[unique_values != 0]) # exclude background

            h, w = label_raw.shape
            tile_size, pad_h, pad_w = compute_tile_size(h, w)

            img_padded = apply_padding(img, pad_h, pad_w)
            mask_padded = apply_padding(mask, pad_h, pad_w)
            label_padded = apply_padding(filled_label, pad_h, pad_w)

            img_tiles = tile_image(img_padded, tile_size)
            mask_tiles = tile_image(mask_padded, tile_size)
            label_tiles = tile_image(label_padded, tile_size)

            base_name = os.path.splitext(os.path.basename(img_path))[0]
            for i, ((im_tile, y, x), (msk_tile, _, _), (lbl_tile, _, _)) in enumerate(zip(img_tiles, mask_tiles, label_tiles)):
                tile_name = f"{base_name}_tile{i}.tif"
                all_tiles.append({
                    'Dataset': dataset,
                    'ImageName': base_name,
                    'TileName': tile_name,
                    'ImagePath': img_path,
                    'TileData': (im_tile, msk_tile, lbl_tile),
                    'Height': im_tile.shape[0],
                    'Width': im_tile.shape[1]
                })

            summary_records.append({
                'Dataset': dataset,
                'ImageName': base_name,
                'Height': h,
                'Width': w,
                'NumFibers': num_fibers,
                'NumTiles': len(img_tiles)
            })

    print(f"Finished tiling all images. Total tiles: {len(all_tiles)}. Shuffling and splitting...")

    # Shuffle all tiles
    random.shuffle(all_tiles)

    # Split tiles
    total_tiles = len(all_tiles)
    n_train = round(total_tiles * 0.8)
    n_val = round(total_tiles * 0.1)
    n_test = total_tiles - n_train - n_val
    split_assignments = ['train'] * n_train + ['val'] * n_val + ['test'] * n_test

    for tile_info, split in zip(all_tiles, split_assignments):
        im_tile, msk_tile, lbl_tile = tile_info['TileData']
        tile_name = tile_info['TileName']
        dataset = tile_info['Dataset']

        out_img_path = os.path.join(out_root, f"{split}_images")
        out_mask_path = os.path.join(out_root, f"{split}_masks")
        out_label_path = os.path.join(out_root, f"{split}_labels")
        os.makedirs(out_img_path, exist_ok=True)
        os.makedirs(out_mask_path, exist_ok=True)
        os.makedirs(out_label_path, exist_ok=True)

        io.imsave(os.path.join(out_img_path, tile_name), im_tile, check_contrast=False)
        io.imsave(os.path.join(out_mask_path, tile_name), msk_tile, check_contrast=False)
        io.imsave(os.path.join(out_label_path, tile_name), lbl_tile.astype(np.uint16), check_contrast=False)

        tile_records.append({
            'Split': split,
            'Dataset': dataset,
            'ImageName': tile_info['ImageName'],
            'TileName': tile_name,
            'Height': tile_info['Height'],
            'Width': tile_info['Width']
        })

    print("Creating summary tables...")

    # Update summary with tile distribution per split
    summary_df = pd.DataFrame(summary_records)
    tile_df = pd.DataFrame(tile_records)

    tile_counts = tile_df.groupby(['Dataset', 'ImageName', 'Split']).size().unstack(fill_value=0).reset_index()
    summary_df = summary_df.merge(tile_counts, on=['Dataset', 'ImageName'], how='left')
    summary_df.rename(columns={
        'train': 'TrainSplit',
        'val': 'ValSplit',
        'test': 'TestSplit'
    }, inplace=True)

    # Save summary TSV
    summary_path = os.path.join(out_root, "dataset_summary.tsv")
    tile_list_path = os.path.join(out_root, "tile_list.tsv")
    summary_df.to_csv(summary_path, sep='\t', index=False)
    tile_df.to_csv(tile_list_path, sep='\t', index=False)

    print(f"\n Done! Saved:\n  - Summary: {summary_path}\n  - Tile list: {tile_list_path}\n  - Total tiles: {len(tile_df)} from {len(summary_df)} images.\n")

def main():
    parser = argparse.ArgumentParser(description="Prepare U-Net training data from datasets.")
    parser.add_argument('--input', type=str, default="datasets", help='Input root folder with datasets')
    parser.add_argument('--output', type=str, default="prepared_data", help='Output root folder')
    args = parser.parse_args()

    create_train_val_test_split_all(args.input, args.output)

if __name__ == "__main__":
    main()
