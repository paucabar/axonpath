import numpy as np
from numpy import ndarray
import edt
from utils.image_processing import fill_labels
from skimage.measure import regionprops, label
from skimage.segmentation import expand_labels
from skimage.morphology import skeletonize, remove_small_objects, remove_small_holes
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import watershed
from skimage.filters import sobel
import line_profiler

def timer(func):
    import functools
    from line_profiler import LineProfiler
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        lp = LineProfiler()
        lp_wrapper = lp(func)
        lp_wrapper(*args, **kwargs)
        lp.print_stats()
        value = func(*args, **kwargs)
        return value
 
    return wrapper

# Erosion method to quickly process labelled images
def erode_labels(image: ndarray):
    edges = sobel(image) != 0
    image_eroded = np.where(edges == False, image, 0)
    return image_eroded

# Calculate boundary distance transform from labelled image
def boundary_dist_trans(image: ndarray):
    eroded = erode_labels(image)
    dist_map = distance_transform_edt(eroded)
    return dist_map

# Calculate skeleton distance transforms from a specific label
def stack_sdt(image_stack: ndarray, skeleton: ndarray):   
    skeleton_stack = image_stack * skeleton # creates a binary image containing only the specified skeleton
    skeleton_stack_inverted = 1 - skeleton_stack # invert skeleton
    for z in range(skeleton_stack_inverted.shape[0]):
        slice = skeleton_stack_inverted[z, :, :]
        slice_distance_transform = distance_transform_edt(slice)
        skeleton_stack_inverted[z, :, :] = slice_distance_transform
    distmap = np.where(image_stack, skeleton_stack_inverted, 0) # keep only the distance map for the label
    return distmap

# get a skeleton distance transforms from all the masks contained in a label image
def skeleton_dist_trans(image: ndarray):
    eroded = erode_labels(image) # erode label image
    eroded = label(eroded) # relabel to have integers
    skeleton = skeletonize(eroded > 0, method='lee') # skeletonize binary image
    skeleton = skeleton > 0 # make sure it's binary (lee method returns an 8-bit image with 0 (False) and 255 (True)))
    
    binary_list = [] # creates empty list to save images of individual distance transforms
    regions = regionprops(eroded)

    # gets the image transform of every label and stores them as individual images
    for i in range(len(regions)):
        label_id = regions[i].label # gets the label id of the current region
        binary_label_id = np.where(image == label_id, 1, 0) # crates binary image containing only the specified label
        binary_list.append(binary_label_id) # stores the image on the list
    binary_stack = np.stack(binary_list) # creates stack from list of images (numpy arrays)

    # generates a new image containing all the distance transforms 
    dist_trans_stack = stack_sdt(binary_stack, skeleton) # creates image containing only the specified distance transform
    image_dist_trans = np.max(dist_trans_stack, axis = 0) # calculates the maximum projection to get back a 2D image
    return image_dist_trans

"""
Distance transform methods using GPU
Uses edt library
"""

# Calculate boundary distance transform from labelled image
def boundary_dist_trans_2(image: ndarray):
# Compute the Euclidean distance transform for the labeled image
    dt = edt.edt(
        image,
        anisotropy=(1, 1), # Assuming equal spacing in x and y directions
        black_border=True,
        order='C', # 'C' (C-order, XYZ, row-major) and 'F' (Fortran-order, ZYX, column major)
        parallel=2 # Number of threads, <= 0 sets to num CPU
    )
    return dt

# Function to get skeleton for each unique label
def get_skeleton_for_labels(image: ndarray):
    unique_labels = np.unique(image)
    skeleton = np.zeros_like(image, dtype=bool)
    for label in unique_labels:
        if label == 0:
            continue  # Skip background
        label_mask = (image == label)
        skeletonized_label = skeletonize(label_mask)
        skeleton[label_mask] = skeletonized_label[label_mask]
    return skeleton

def skeleton_dist_trans_2(image: ndarray, skeleton: ndarray):
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
        dt_bbox = edt.edt(inverted_skeleton_bbox, black_border=False, order='C', parallel=1)
        
        # Place the computed distance transform values into the output image
        dt_within_labels[min_row:max_row, min_col:max_col][label_bbox] = dt_bbox[label_bbox]
    
    return dt_within_labels


"""
Safely divide two images element-wise while handling division by zero.

Args:
    image1 (numpy.ndarray): The first image (numpy array).
    image2 (numpy.ndarray): The second image (numpy array).

Returns:
    numpy.ndarray: The result of the division with handling division by zero.
"""

def safe_divide_images(image1, image2):
    # Check for division by zero and set the result to zero where it occurs
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.true_divide(image1, image2)
        result[~np.isfinite(result)] = 0  # Handle division by zero and NaN cases

    return result

def skeleton_aware_dist_trans(image: ndarray, alpha: float):
    bdt = boundary_dist_trans_2(image)
    sdt = skeleton_dist_trans_2(image)
    sadt_function = safe_divide_images(bdt, bdt + sdt) ** alpha
    return sadt_function




class LabelDistanceTransforms:
    def __init__(self, label_image: ndarray, alpha: float = 0.8, fill_gt: bool = True, background_transform: bool = False, signed_background = False):
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