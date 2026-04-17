import numpy as np
import edt
from numpy import ndarray
from skimage.measure import regionprops, label
from skimage.morphology import skeletonize
from scipy.ndimage import binary_fill_holes


class LabelDistanceTransforms:
    def __init__(
        self,
        label_image: ndarray,
        alpha: float = 0.3,
        fill_gt: bool = False,
        background_transform: bool = False,
        signed_background: bool = False
    ):
        """
        Initialize the LabelDistanceTransforms class.

        Args:
            label_image (ndarray): 2D array of labeled regions (0 is background).
            alpha (float): Exponent for soft skeleton-aware transform.
            fill_gt (bool): Whether to fill holes in label masks.
            background_transform (bool): Whether to subtract background soft transform.
            signed_background (bool): Whether to set background to -1.
        """
        if label_image.ndim != 2:
            raise ValueError("label_image must be a 2D array.")
        if not np.issubdtype(label_image.dtype, np.integer):
            raise TypeError("label_image must contain integers (label IDs).")

        self.label_image = label_image
        self.alpha = alpha
        self.fill_gt = fill_gt
        self.background_transform = background_transform
        self.signed_background = signed_background

        self.regions = regionprops(label_image.astype(np.int32))
        self.region_lookup = {r.label: r for r in self.regions}

    def __safe_divide_images(self, image1: ndarray, image2: ndarray):
        """Safely divides two images, avoiding division by zero."""
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.true_divide(image1, image2)
            result[~np.isfinite(result)] = 0
        return result

    def _fill_mask(self, mask: ndarray):
        """Fill holes in binary mask using full area."""
        return binary_fill_holes(mask)

    def _compute_skeleton(self, mask: ndarray):
        """Skeletonize a binary mask."""
        skeleton = skeletonize(mask, method='lee')
        return skeleton

    def _compute_distance_transforms(self, image: ndarray):
        """
        Compute boundary and skeleton distance transforms for each label
        using bounding boxes for efficiency.
        """
        boundary_dt = np.zeros_like(image, dtype=float)
        skel_dist = np.zeros_like(image, dtype=float)

        for region in self.regions:
            label_id = region.label
            min_row, min_col, max_row, max_col = region.bbox

            label_mask = (image == label_id)
            cropped_mask = label_mask[min_row:max_row, min_col:max_col]

            if self.fill_gt:
                cropped_mask = self._fill_mask(cropped_mask)

            # Boundary distance transform — pad with False so objects that fill
            # their bbox (e.g. perfect rectangles) don't produce inf values
            padded_mask = np.pad(cropped_mask, 1, mode='constant', constant_values=False)
            bdt_padded = edt.edt(padded_mask, black_border=False, parallel=2)
            bdt_crop = bdt_padded[1:-1, 1:-1]
            boundary_dt[min_row:max_row, min_col:max_col][cropped_mask] = bdt_crop[cropped_mask]

            # Distance to skeleton
            skeleton_crop = self._compute_skeleton(cropped_mask)
            skel_dist_crop = edt.edt(~skeleton_crop, black_border=False, parallel=2)
            skel_dist[min_row:max_row, min_col:max_col][cropped_mask] = (skel_dist_crop * cropped_mask)[cropped_mask]

        return boundary_dt, skel_dist

    def _compute_background_transforms(self, image: ndarray):
        """
        Compute soft-aware distance transform on background as if it were labeled.
        """
        background_mask = (image == 0)
        labeled_background = label(background_mask.astype(np.uint8))
        bdt, skel_dist = self._compute_distance_transforms(labeled_background)
        sadt_background = self.__safe_divide_images(bdt, bdt + skel_dist) ** self.alpha
        return sadt_background

    def _get_background_mask(self, image: ndarray):
        return (image == 0)

    def skeleton_aware_dist_trans(self):
        """
        Compute the skeleton-aware distance transform (SDT).

        Returns:
            sdt (ndarray): Skeleton-aware distance transform. Values in [0, 1]
                inside objects (0 at boundary, 1 at skeleton), -1 in background
                when signed_background=True.
            bdt (ndarray): Boundary distance transform (EDT from each pixel to
                the nearest object boundary).
            skel_dist (ndarray): Distance-to-skeleton transform (EDT from each
                pixel to the nearest skeleton pixel).
        """
        bdt, skel_dist = self._compute_distance_transforms(self.label_image)
        sdt = self.__safe_divide_images(bdt, bdt + skel_dist) ** self.alpha

        if self.background_transform:
            sadt_background = self._compute_background_transforms(self.label_image)
            sdt = sdt - sadt_background

        if self.signed_background:
            sdt[self._get_background_mask(self.label_image)] = -1

        return sdt, bdt, skel_dist