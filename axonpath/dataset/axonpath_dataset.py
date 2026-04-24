import os
import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
from glob import glob
from axonpath.utils.image_processing import normalize

_EXPECTED_TILE_SIZE = (512, 512)


class AxonPathDataset(Dataset):
    def __init__(self, tile_dir, transform=None, cache=False, _skip_size_check=False):
        self.tile_paths = sorted(glob(os.path.join(tile_dir, "*.npy")))
        print(f"Discovered {len(self.tile_paths)} .npy tiles in '{tile_dir}'")
        self.transform = transform
        self.cache = cache
        self._data_cache = {} if cache else None
        self._skip_size_check = _skip_size_check

    def populate_cache(self):
        """Preload all .npy tiles into memory to avoid loading during training."""
        if not self.cache:
            raise RuntimeError("Dataset was not initialized with cache=True")

        for path in tqdm(self.tile_paths, desc="Populating cache"):
            if path not in self._data_cache:
                try:
                    raw = np.load(path, allow_pickle=True).item()
                    h, w = raw["image"].shape[:2]
                    if not self._skip_size_check and (h < _EXPECTED_TILE_SIZE[0] or w < _EXPECTED_TILE_SIZE[1]):
                        raise ValueError(
                            f"Tile '{os.path.basename(path)}' has size {h}x{w}, "
                            f"which is smaller than the expected minimum "
                            f"{_EXPECTED_TILE_SIZE[0]}x{_EXPECTED_TILE_SIZE[1]}. "
                            "Re-prepare the dataset."
                        )
                    # Normalize once and store the result to avoid recomputing each epoch
                    data = dict(raw)
                    data["image"] = normalize(raw["image"]).astype(np.float32)
                    self._data_cache[path] = data
                except Exception as e:
                    print(f"Error loading {path}: {e}")
                    raise RuntimeError(f"Failed to cache {path}") from e

    def __getitem__(self, idx):
        path = self.tile_paths[idx]

        # Load from cache or disk
        if self.cache and path in self._data_cache:
            data = self._data_cache[path]
        else:
            try:
                raw = np.load(path, allow_pickle=True).item()
                if self.cache:
                    data = dict(raw)
                    data["image"] = normalize(raw["image"]).astype(np.float32)
                    self._data_cache[path] = data
                else:
                    data = raw
            except Exception as e:
                print(f"Error loading .npy file at {path}: {e}")
                raise RuntimeError(f"Corrupt .npy: {path}") from e

        try:
            image = data["image"].astype(np.float32) if self.cache else normalize(data["image"]).astype(np.float32)

            mask_sem = np.where(data["mask_sem"] == 3, 2, data["mask_sem"])

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
                    raise RuntimeError(f"Transform failed on {path}") from e

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

    def __str__(self):
        if self.cache:
            return f"{len(self._data_cache)} image sets cached / {len(self.tile_paths)} total"
        else:
            return f"Caching disabled — {len(self.tile_paths)} image sets available"
