import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
from skimage.measure import label
from glob import glob
from utils.image_processing import normalize, normalize_saturated, fill_labels
from skeleton.skeleton_aware_distance_transform import LabelDistanceTransforms

class AimSegDataset(Dataset):
    def __init__(self, tile_dir, transform=None):
        self.tile_paths = sorted(glob(os.path.join(tile_dir, "*.npy")))
        print(f"Loaded {len(self.tile_paths)} .npy tiles from {tile_dir}")
        self.transform = transform

    def __getitem__(self, idx):
        path = self.tile_paths[idx]

        try:
            data = np.load(path, allow_pickle=True).item()
        except Exception as e:
            print(f"Error loading .npy file at {path}: {e}")
            raise RuntimeError(f"Corrupt .npy: {path}") from e

        try:
            image = normalize_saturated(data["image"]).astype(np.float32)

            mask_sem = data["mask_sem"]
            np.putmask(mask_sem, mask_sem == 3, 2)

            masks = [
                data["mask_fibre"],
                data["mask_axon"],
                mask_sem,
                data["sdt_fibre"],
                data["sdt_axon"]
            ]

            if self.transform:
                try:
                    transformed = self.transform(image=image, masks=masks)
                    image = transformed["image"]
                    masks = transformed["masks"]
                except Exception as e:
                    print(f"[Worker {os.getpid()}] Transform failed: {e}")
                    raise RuntimeError(f"Albumentations transform failed on {path}") from e

            # Robust conversion to torch.Tensor
            image_tensor = image if isinstance(image, torch.Tensor) else torch.from_numpy(image)
            masks_tensor = [
                m if isinstance(m, torch.Tensor) else torch.from_numpy(m.astype(np.float32))
                for m in masks
            ]

            return image_tensor, masks_tensor

        except Exception as e:
            print(f"Transform or tensor conversion failed on {path}: {e}")
            raise RuntimeError(f"Failed to transform or convert data from {path}") from e


    def __len__(self):
        return len(self.tile_paths)


class AimSegDatasetOld(Dataset):
    def __init__(self, image_dir, masksem_dir, maskins_dir, transform=None):
        self.image_dir = image_dir
        self.masksem_dir = masksem_dir
        self.maskins_dir = maskins_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self._cache = {}

    def __str__(self):
        return f"{len(self._cache) / 3} image sets in cache"
    
    def __len__(self):
        return len(self.images)
    
    def _load_image(self, image_path: str):
        image = np.array(Image.open(image_path)).astype(np.float32)
        image = image[:, :] # alternatively use [:-1, :-1] since sometimes masks are disconnected from image edges after downampling
        if (os.path.abspath(image_path).startswith(os.path.abspath(self.image_dir))):
            image = normalize(image) # normalize the raw data
            return image
        if (os.path.abspath(image_path).startswith(os.path.abspath(self.maskins_dir))):
            distance_transform, __, __ = LabelDistanceTransforms(image, 0.3, True, False, True).skeleton_aware_dist_trans() # calculate distance transform
            return [image, distance_transform] # list containing both instance mask and distance transform
        if (os.path.abspath(image_path).startswith(os.path.abspath(self.masksem_dir))):
            axon_mask = (image == 3).astype(np.uint8)  # Create a binary mask for the original label 3
            axon_labels = label(axon_mask) # Label connected components in the axon mask
            np.putmask(image, image == 3, 2) # Safely merge labels: Convert label 3 to label 2
            distance_transform, _, _ = LabelDistanceTransforms(axon_labels, 0.1, True, False, True).skeleton_aware_dist_trans() # Calculate the distance transform on the labeled axons
            return [image, axon_labels, distance_transform] # Return a list containing the semantic mask and the axon distance transform
    
    def _get_image(self, image_path: str):
        image = self._cache.get(image_path)
        if image is None:
            image = self._load_image(image_path)
            self._cache[image_path] = image
        return image.copy() # Copy to avoid overwriting by augmentations

    
    def _get_path(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        masksem_path = os.path.join(self.masksem_dir, self.images[index])
        maskins_path = os.path.join(self.maskins_dir, self.images[index])
        return img_path, masksem_path, maskins_path

    def populate_cache(self):
        for index in range(0, len(self.images)):
            img_path, masksem_path, maskins_path = self._get_path(index)
            image = self._get_image(img_path)
            image = self._get_image(masksem_path)
            image = self._get_image(maskins_path)
        return

    def __getitem__(self, index):
        img_path, masksem_path, maskins_path = self._get_path(index)
        # open image
        image = self._get_image(img_path)
        # open semantic mask and axon distance transform
        mask_sem_list = self._get_image(masksem_path)
        mask_sem = mask_sem_list[0]
        mask_ins_axon = mask_sem_list[1]
        mask_ins_axon = fill_labels(mask_ins_axon)
        axon_distance_transform = mask_sem_list[2]
        # open fibre instance mask and fibre distance transform
        mask_ins_list = self._get_image(maskins_path)
        mask_ins_fibre = mask_ins_list[0]
        mask_ins_fibre = fill_labels(mask_ins_fibre)
        fibre_distance_transform = mask_ins_list[1]
        # masks to list
        masks = [mask_ins_fibre, mask_ins_axon, mask_sem, fibre_distance_transform, axon_distance_transform]
        
        #transforms
        if self.transform is not None:
            augmentations = self.transform(image=image, masks=masks)
            image = augmentations["image"]
            masks = augmentations["masks"]

        return image, masks
