import os
from skimage import io
from glob import glob
from shutil import copyfile
import random
import time
from image_processing import (
    fill_labels,
)

def shuffle_tuples_in_list(list1, list2, list3):
    assert len(list1) == len(list2) == len(list3)
    list_of_tuples = list(zip(list1, list2, list3))
    random.shuffle(list_of_tuples)
    list1, list2, list3 = zip(*list_of_tuples)
    return list1, list2, list3

def create_train_val_test_split(in_folder, out_folder):
    print(in_folder)
    image_paths = glob(os.path.join(in_folder, "images", "*.tif"))
    image_paths.sort()
    mask_paths = glob(os.path.join(in_folder, "masks", "*.tif"))
    mask_paths.sort()
    label_paths = glob(os.path.join(in_folder, "labels", "*.tif"))
    label_paths.sort()
    assert len(image_paths) == len(mask_paths) == len(label_paths)

    # create out folder
    os.makedirs(out_folder, exist_ok=True)

    image_paths, mask_paths, label_paths = shuffle_tuples_in_list(image_paths, mask_paths, label_paths)
    print(list(zip(image_paths, mask_paths, label_paths)))

    # % to split dataset
    perc_train = 0.8
    perc_val = 0.2
    n_train = round(len(image_paths) * perc_train)
    n_val = round(len(image_paths) * perc_val)

    for ii, (image, mask, label) in enumerate(zip(image_paths, mask_paths, label_paths)):
        if ii < n_train:
            split = "train"
        elif ii < (n_train + n_val):
            split = "val"
        else:
            split = "test"      
        image_out = os.path.join(out_folder, split + "_images")
        mask_out = os.path.join(out_folder, split + "_masks")
        label_out = os.path.join(out_folder, split + "_labels")
        os.makedirs(image_out, exist_ok=True)
        os.makedirs(mask_out, exist_ok=True)
        os.makedirs(label_out, exist_ok=True)
        copyfile(image, os.path.join(image_out, os.path.split(image)[1]))
        copyfile(mask, os.path.join(mask_out, os.path.split(mask)[1]))
        #copyfile(label, os.path.join(label_out, os.path.split(label)[1]))

        # fill holes on instance labels
        start = time.time()
        label_image = io.imread(label)
        filled_label = fill_labels(label_image)
        end = time.time()
        print(f"Elapsed time for {os.path.split(image)[1]}: {end - start} seconds")
        #filled_label = set_glasbey_cmap(filled_label)
        io.imsave(os.path.join(label_out, os.path.split(label)[1]), filled_label, check_contrast=False)

def main():
    create_train_val_test_split("data_pre", "data")


if __name__ == "__main__":
    main()