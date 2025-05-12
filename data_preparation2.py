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
    return list1, list2, list3

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
            tiles.append(image[y:y+tile_size, x:x+tile_size])
    return tiles

def apply_padding(im, pad_h, pad_w):
    if im.ndim == 3:
        return np.pad(im, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
    else:
        return np.pad(im, ((0, pad_h), (0, pad_w)), mode='constant')

def split_dataset(image_paths, mask_paths, label_paths):
    image_paths, mask_paths, label_paths = shuffle_tuples_in_list(image_paths, mask_paths, label_paths)
    n_total = len(image_paths)
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
    splits = ['train'] * n_train + ['val'] * n_val + ['test'] * n_test
    return list(zip(image_paths, mask_paths, label_paths, splits))

def create_train_val_test_split_all(in_root, out_root):
    os.makedirs(out_root, exist_ok=True)
    csv_records = []
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

        split_info = split_dataset(image_paths, mask_paths, label_paths)

        for img_path, msk_path, lbl_path, split in split_info:
            img = io.imread(img_path)
            mask = io.imread(msk_path)
            label_raw = io.imread(lbl_path)

            filled_label = fill_labels(label_raw.astype(np.int32))
            num_fibers = len(np.unique(filled_label)) - 1  # exclude background

            h, w = label_raw.shape
            tile_size, pad_h, pad_w = compute_tile_size(h, w)

            img_padded = apply_padding(img, pad_h, pad_w)
            mask_padded = apply_padding(mask, pad_h, pad_w)
            label_padded = apply_padding(filled_label, pad_h, pad_w)

            img_tiles = tile_image(img_padded, tile_size)
            mask_tiles = tile_image(mask_padded, tile_size)
            label_tiles = tile_image(label_padded, tile_size)

            base_name = os.path.splitext(os.path.basename(img_path))[0]
            for i, (im_tile, msk_tile, lbl_tile) in enumerate(zip(img_tiles, mask_tiles, label_tiles)):
                tile_name = f"{dataset}_{base_name}_tile{i}.tif"

                out_img_path = os.path.join(out_root, f"{split}_images")
                out_mask_path = os.path.join(out_root, f"{split}_masks")
                out_label_path = os.path.join(out_root, f"{split}_labels")
                os.makedirs(out_img_path, exist_ok=True)
                os.makedirs(out_mask_path, exist_ok=True)
                os.makedirs(out_label_path, exist_ok=True)

                io.imsave(os.path.join(out_img_path, tile_name), im_tile, check_contrast=False)
                io.imsave(os.path.join(out_mask_path, tile_name), msk_tile, check_contrast=False)
                io.imsave(os.path.join(out_label_path, tile_name), lbl_tile.astype(np.uint16), check_contrast=False)

            rel_path = os.path.relpath(img_path, start=in_root)
            print(f"Processed {rel_path} -> {len(img_tiles)} tiles")

            csv_records.append([dataset, base_name, h, w, num_fibers, len(img_tiles), split])

    # Save CSV summary
    csv_path = os.path.join(out_root, "dataset_summary.csv")
    with open(csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter='\t')
        writer.writerow(['Dataset', 'ImageName', 'Height', 'Width', 'NumFibers', 'NumTiles', 'Split'])
        writer.writerows(csv_records)

def main():
    parser = argparse.ArgumentParser(description="Prepare U-Net training data from datasets.")
    parser.add_argument('--input', type=str, default="datasets", help='Input root folder with datasets')
    parser.add_argument('--output', type=str, default="prepared_data", help='Output root folder')
    args = parser.parse_args()

    create_train_val_test_split_all(args.input, args.output)

if __name__ == "__main__":
    main()
