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
        return skeleton & mask  # Ensure it stays within mask

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
            inv_skel_crop = ~skeleton_crop & cropped_mask
            sdt_crop = edt.edt(inv_skel_crop, black_border=False, parallel=2)
            skeleton_dt[min_row:max_row, min_col:max_col][cropped_mask] = sdt_crop[cropped_mask]

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



class LabelDistanceTransformsOld:
    def __init__(self, label_image: ndarray, alpha: float = 0.3, fill_gt: bool = True, background_transform: bool = False, signed_background = False):
        self.label_image = fill_labels(label_image)
        self.alpha = alpha
        self.fill_gt = fill_gt
        self.regions = regionprops(label_image.astype(np.int16))
        self.background_transform = background_transform
        self.signed_background = signed_background

    def __safe_divide_images(self, image1: ndarray, image2: ndarray):
        """
        Safely divide two images element-wise while handling division by zero.

        Args:
            image1 (numpy.ndarray): The first image (numpy array).
            image2 (numpy.ndarray): The second image (numpy array).

        Returns:
            numpy.ndarray: The result of the division with handling division by zero.
        """

        # Check for division by zero and set the result to zero where it occurs
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.true_divide(image1, image2)
            result[~np.isfinite(result)] = 0  # Handle division by zero and NaN cases
        return result

    def _get_binary_stack(self, image: ndarray):
        binary_list = [] # creates empty list to save images of individual distance transforms
        # gets the image mask of every label and and creates a stack
        for i in range(len(self.regions)):
            label_id = self.regions[i].label # gets the label id of the current region
            binary_label_id = np.where(image == label_id, 1, 0) # crates binary image containing only the specified label
            binary_list.append(binary_label_id) # stores the binary image on the list
        binary_stack = np.stack(binary_list) # creates stack from list of images
        return binary_stack

    def _fill_mask(self, image: ndarray, index):   
        return remove_small_holes(image, self.regions[index].area) # fills the binary mask

    def _get_skeleton(self, image: ndarray, method: str = 'lee'): # or 'zhang'
        skeleton = skeletonize(image > 0, method=method) # skeletonize binary image
        skeleton = skeleton > 0 # make sure it's binary (lee method returns an 8-bit image with 0 (False) and 255 (True)))
        # TODO: prune skeleton?
        return skeleton

    def _transform_stack(self, image_stack: ndarray):   
        boundary_dt_stack = np.zeros_like(image_stack)
        skeleton_dt_stack = np.zeros_like(image_stack)
        for z in range(image_stack.shape[0]):
            if(self.fill_gt):
                image_stack[z, :, :] = self._fill_mask(image_stack[z, :, :], z)
            slice_skeleton = self._get_skeleton(image_stack[z, :, :], 'lee')
            boundary_dt_stack[z, :, :] = distance_transform_edt(image_stack[z, :, :])
            skeleton_dt_stack[z, :, :] = distance_transform_edt(1 - slice_skeleton)
        boundary_dt_stack = np.where(image_stack, boundary_dt_stack, 0)
        skeleton_dt_stack = np.where(image_stack, skeleton_dt_stack, 0)
        return boundary_dt_stack, skeleton_dt_stack

    """
    New distance transform methods
    """

    # Calculate boundary distance transform from labelled image
    def _boundary_dist_trans(self, image: ndarray):
    # Compute the Euclidean distance transform for the labeled image
        dt = edt.edt(
            image,
            anisotropy=(1, 1), # Assuming equal spacing in x and y directions
            black_border=False,
            order='C', # 'C' (C-order, XYZ, row-major) and 'F' (Fortran-order, ZYX, column major)
            parallel=2 # Number of threads, <= 0 sets to num CPU
        )
        return dt

    # Function to get skeleton for each unique label
    def _get_skeleton_for_labels(self, image: ndarray, method: str = 'lee'): # 'lee' or 'zhang'
        unique_labels = np.unique(image)
        skeleton = np.zeros_like(image, dtype=bool)
        for label in unique_labels:
            if label == 0:
                continue  # Skip background
            label_mask = (image == label)
            skeletonized_label = skeletonize(label_mask, method=method)
            skeleton[label_mask] = skeletonized_label[label_mask]
        return skeleton

    def _skeleton_dist_trans(self, image: ndarray, skeleton: ndarray):
        unique_labels = np.unique(image)
        dt_within_labels = np.zeros_like(image, dtype=float)
        
        for label in unique_labels:
            if label == 0:
                continue  # Skip background
            
            # Create a mask for the current label
            label_mask = (image == label)
            
            # Get the bounding box coordinates for the current label
            region_props = regionprops(label_mask.astype(int))
            bbox = region_props[0].bbox  # Get bounding box coordinates
            min_row, min_col, max_row, max_col = bbox
            
            # Extract the region of interest for the bounding box
            label_bbox = label_mask[min_row:max_row, min_col:max_col]
            skeleton_bbox = skeleton[min_row:max_row, min_col:max_col] & label_bbox
            
            # Invert the skeleton within the bounding box
            inverted_skeleton_bbox = ~skeleton_bbox
            
            # Compute the distance transform within the bounding box
            dt_bbox = edt.edt(
                inverted_skeleton_bbox,
                black_border=False,
                order='C',
                parallel=2
                )
            
            # Place the computed distance transform values into the output image
            dt_within_labels[min_row:max_row, min_col:max_col][label_bbox] = dt_bbox[label_bbox]
        
        return dt_within_labels


    def _get_distance_transforms(self, image: ndarray):
        #binary_stack = self._get_binary_stack(image)
        #boundary_dt_stack, skeleton_dt_stack = self._transform_stack(binary_stack) # creates image containing only the specified distance transform
        #boundary_distance_transform = np.max(boundary_dt_stack, axis = 0) # calculates the maximum projection to get back a 2D image
        #skeleton_distance_transform = np.max(skeleton_dt_stack, axis = 0) # calculates the maximum projection to get back a 2D image
        boundary_distance_transform = self._boundary_dist_trans(image)
        # Extract the skeleton for each label
        skeleton = self._get_skeleton_for_labels(image, 'lee')
        # Compute the distance transform from the inverted skeleton, confined within each label's mask
        skeleton_distance_transform = self._skeleton_dist_trans(image, skeleton)
        return boundary_distance_transform, skeleton_distance_transform

    def _get_background_transforms(self, image: ndarray):
        binary_background = image == 0
        label_background = label(binary_background)
        bdt, sdt = self._get_distance_transforms(label_background)
        sadt_background = self.__safe_divide_images(bdt, bdt + sdt) ** self.alpha
        return sadt_background
    
    def _get_background(self, image: ndarray):
        return image == 0

    def skeleton_aware_dist_trans(self):
        bdt, sdt = self._get_distance_transforms(self.label_image)
        sadt_function = self.__safe_divide_images(bdt, bdt + sdt) ** self.alpha
        if (self.background_transform):
            sadt_background = self._get_background_transforms(self.label_image)
            sadt_function = sadt_function - sadt_background
        if (self.signed_background):
            binary_background = self._get_background(self.label_image)
            sadt_function[binary_background == True] = -1
        return sadt_function, bdt, sdt