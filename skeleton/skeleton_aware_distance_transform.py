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
        skeleton_dt = np.zeros_like(image, dtype=float)

        for region in self.regions:
            label_id = region.label
            min_row, min_col, max_row, max_col = region.bbox

            label_mask = (image == label_id)
            cropped_mask = label_mask[min_row:max_row, min_col:max_col]

            if self.fill_gt:
                cropped_mask = self._fill_mask(cropped_mask)

            # Boundary distance transform
            bdt_crop = edt.edt(cropped_mask, black_border=False, parallel=2)
            boundary_dt[min_row:max_row, min_col:max_col][cropped_mask] = bdt_crop[cropped_mask]

            # Skeleton distance transform
            skeleton_crop = self._compute_skeleton(cropped_mask)
            inv_skel_crop = ~skeleton_crop
            sdt_crop = edt.edt(inv_skel_crop, black_border=False, parallel=2)
            sdt_crop_masked = sdt_crop * cropped_mask
            skeleton_dt[min_row:max_row, min_col:max_col][cropped_mask] = sdt_crop_masked[cropped_mask]


        return boundary_dt, skeleton_dt

    def _compute_background_transforms(self, image: ndarray):
        """
        Compute soft-aware distance transform on background as if it were labeled.
        """
        background_mask = (image == 0)
        labeled_background = label(background_mask.astype(np.uint8))
        bdt, sdt = self._compute_distance_transforms(labeled_background)
        sadt_background = self.__safe_divide_images(bdt, bdt + sdt) ** self.alpha
        return sadt_background

    def _get_background_mask(self, image: ndarray):
        return (image == 0)

    def skeleton_aware_dist_trans(self):
        """
        Compute the skeleton-aware distance transform (SADT) function.
        Returns:
            sadt_function (ndarray): Final skeleton-aware function.
            bdt (ndarray): Boundary distance transform.
            sdt (ndarray): Skeleton distance transform.
        """
        bdt, sdt = self._compute_distance_transforms(self.label_image)
        sadt_function = self.__safe_divide_images(bdt, bdt + sdt) ** self.alpha

        if self.background_transform:
            sadt_background = self._compute_background_transforms(self.label_image)
            sadt_function = sadt_function - sadt_background

        if self.signed_background:
            sadt_function[self._get_background_mask(self.label_image)] = -1

        return sadt_function, bdt, sdt