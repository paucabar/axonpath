import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from skimage.measure import label
from image_processing import normalize, fill_labels
from skeleton_aware_distance_transform import LabelDistanceTransforms

class AimSegDataset(Dataset):
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
        image = image[:-1, :-1] # sometimes masks are disconnected from image edges after downampling
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
            distance_transform, _, _ = LabelDistanceTransforms(axon_labels, 0.3, True, False, True).skeleton_aware_dist_trans() # Calculate the distance transform on the labeled axons
            return [image, distance_transform] # Return a list containing the semantic mask and the axon distance transform
    
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
        axon_distance_transform = mask_sem_list[1]
        # open fibre instance mask and fibre distance transform
        mask_ins_list = self._get_image(maskins_path)
        mask_ins = mask_ins_list[0]
        fibre_distance_transform = mask_ins_list[1]
        # masks to list
        masks = [mask_ins, mask_sem, fibre_distance_transform, axon_distance_transform]
        
        #transforms
        if self.transform is not None:
            augmentations = self.transform(image=image, masks=masks)
            image = augmentations["image"]
            masks = augmentations["masks"]

        return image, masks
