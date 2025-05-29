import os
import argparse
from skimage import io
from glob import glob
import random
import numpy as np
import pandas as pd
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
            if size % ts == 0:
                return ts, 0
        return size, 0

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

def split_tile_list(tiles, val_ratio=0.1, test_ratio=0.1):
    random.shuffle(tiles)
    total = len(tiles)
    n_val = round(total * val_ratio)
    n_test = round(total * test_ratio)
    n_train = total - n_val - n_test
    return (
        tiles[:n_train],
        tiles[n_train:n_train+n_val],
        tiles[n_train+n_val:]
    )

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
            for i, (im_tile, msk_tile, lbl_tile) in enumerate(zip(img_tiles, mask_tiles, label_tiles)):
                all_tiles.append((dataset, base_name, i, im_tile, msk_tile, lbl_tile, tile_h, tile_w))

            summary_records.append([dataset, base_name, h, w, num_fibers, len(img_tiles)])
            print(f"  Tiled {base_name}: {len(img_tiles)} tiles")

        print(f"Splitting {len(all_tiles)} tiles...")
        train_tiles, val_tiles, test_tiles = split_tile_list(all_tiles)
        split_map = [(train_tiles, 'train'), (val_tiles, 'val'), (test_tiles, 'test')]

        for split_tiles, split in split_map:
            for dataset, base_name, idx, im_tile, msk_tile, lbl_tile, tile_h, tile_w in split_tiles:
                tile_name = f"{base_name}_tile{idx}.tif"

                out_img_path = os.path.join(out_root, f"{split}_images")
                out_mask_path = os.path.join(out_root, f"{split}_masks")
                out_label_path = os.path.join(out_root, f"{split}_labels")
                os.makedirs(out_img_path, exist_ok=True)
                os.makedirs(out_mask_path, exist_ok=True)
                os.makedirs(out_label_path, exist_ok=True)

                io.imsave(os.path.join(out_img_path, tile_name), im_tile, check_contrast=False)
                io.imsave(os.path.join(out_mask_path, tile_name), msk_tile, check_contrast=False)
                io.imsave(os.path.join(out_label_path, tile_name), lbl_tile.astype(np.uint16), check_contrast=False)

                tile_records.append([split, tile_name, dataset, tile_h, tile_w])

        print(f"Finished dataset: {dataset}\n")

    # Save summary TSVs
    summary_df = pd.DataFrame(summary_records, columns=['Dataset', 'ImageName', 'Height', 'Width', 'NumFibers', 'NumTiles'])
    summary_df['TrainSplit'] = int(len(train_tiles))
    summary_df['ValSplit'] = int(len(val_tiles))
    summary_df['TestSplit'] = int(len(test_tiles))
    summary_path = os.path.join(out_root, "dataset_summary.tsv")
    summary_df.to_csv(summary_path, sep='\t', index=False)

    tile_df = pd.DataFrame(tile_records, columns=['Split', 'ImageName', 'Dataset', 'Height', 'Width'])
    tile_list_path = os.path.join(out_root, "tiles_info.tsv")
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
