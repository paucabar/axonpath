import torch
from scipy.ndimage import distance_transform_edt, binary_fill_holes
from skimage.measure import regionprops, label
from skimage.segmentation import expand_labels, watershed
from skimage.morphology import remove_small_objects, skeletonize, remove_small_holes
from skimage.filters import sobel
import numpy as np
import cv2 as cv
import torch.nn.functional as F

from matplotlib.colors import LinearSegmentedColormap
import colorcet as cc

def normalize(image):
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    return image

def normalize_saturated(image, low_perc=1, high_perc=99):
    
    # Calculate lower and upper percentile values
    lower_bound = np.percentile(image, low_perc)
    upper_bound = np.percentile(image, high_perc)
    
    # Clip the image to the lower and upper bounds
    image_clipped = np.clip(image, lower_bound, upper_bound)
    
    # Normalize the image to the range [0, 1] after clipping
    normalized_image = (image_clipped - lower_bound) / (upper_bound - lower_bound)
    
    return normalized_image

# Methods to fill holes on label image
# Converts a single label into a binary mask and fills its holes
def fill_mask(image, label_id, min_area):   
    binary_label_id = (image == label_id)  # creates boolean mask directly
    filled = remove_small_holes(binary_label_id, min_area)  # fills the binary mask
    filled_label_id = np.where(filled, label_id, 0)  # use boolean indexing

    return filled_label_id



def fill_border_holes(mask: np.ndarray) -> np.ndarray:
    """
    Python adaptation of the 'Fill_Border_Holes' ImageJ macro by G. Landini.
    Fills holes in binary masks, including those that touch at most two borders 
    (i.e., corners or edges), mimicking the macro's logic using edge manipulation.

    Original macro by G. Landini:
    https://sites.imagej.net/Landini/plugins/Morphology/Fill_Border_Holes.ijm-20140627120652

    Args:
        mask (np.ndarray): Binary mask (2D boolean or 0/1 array).

    Returns:
        np.ndarray: Mask with internal and edge-connected holes filled.
    """
    if mask.ndim != 2:
        raise ValueError("Only 2D arrays supported.")

    h, w = mask.shape
    padded = np.pad(mask.astype(bool), 1, mode='constant', constant_values=0)
    filled = padded.copy()

    # Step 1: Right and top borders set to 1
    filled[0, :] = True           # top row
    filled[:, -1] = True          # right column
    filled = binary_fill_holes(filled)

    # Step 2: Reset top to 0, bottom to 1
    filled[0, :] = False
    filled[-1, :] = True
    filled = binary_fill_holes(filled)

    # Step 3: Reset right to 0, left to 1
    filled[:, -1] = False
    filled[:, 0] = True
    filled = binary_fill_holes(filled)

    # Step 4: Reset top to 1, bottom to 0
    filled[0, :] = True
    filled[-1, :] = False
    filled = binary_fill_holes(filled)

    # Step 5: Remove padding
    result = filled[1:-1, 1:-1]

    return result.astype(bool)


def fill_labels(label_image: np.ndarray) -> np.ndarray:
    """
    Fill holes in each labeled region, including edge-touching enclosed holes,
    using bounding box cropping for efficiency.

    Args:
        label_image (np.ndarray): 2D labeled image with integer labels.

    Returns:
        np.ndarray: Labeled image with holes filled per instance.
    """
    if label_image.ndim != 2:
        raise ValueError("label_image must be a 2D array.")
    if not np.issubdtype(label_image.dtype, np.integer):
        raise TypeError("label_image must contain integer labels.")

    filled_image = np.copy(label_image)
    height, width = label_image.shape

    regions = regionprops(label_image.astype(np.int32))

    for region in regions:
        label_id = region.label
        min_row, min_col, max_row, max_col = region.bbox

        # Add 1-pixel padding, but stay within image bounds
        pad_top = 1 if min_row > 0 else 0
        pad_bottom = 1 if max_row < height else 0
        pad_left = 1 if min_col > 0 else 0
        pad_right = 1 if max_col < width else 0

        pad_min_row = min_row - pad_top
        pad_max_row = max_row + pad_bottom
        pad_min_col = min_col - pad_left
        pad_max_col = max_col + pad_right

        cropped_label = label_image[pad_min_row:pad_max_row, pad_min_col:pad_max_col]
        region_mask = (cropped_label == label_id)

        # Fill internal and edge-touching holes
        filled_region_mask = fill_border_holes(region_mask)

        # Assign newly filled pixels back to output image
        filled_image[pad_min_row:pad_max_row, pad_min_col:pad_max_col][filled_region_mask] = label_id


    return filled_image






# last layers for semantic segmentation
def last_layer_fn(pred: torch.Tensor):
    m = torch.nn.Softmax(dim=1)
    smax = m(pred)
    argmax = torch.argmax(smax, dim=1)  # get class predictions
    return argmax



def last_layer_fn_torchscript(pred: torch.Tensor):
    smax = F.softmax(pred, dim = 1)
    argmax = torch.argmax(smax, dim=1)
    return argmax

# post-processing for distance transform
def postprocessing_fn(sem_out: torch.Tensor):
    np_arr = sem_out.squeeze().detach().cpu().numpy()
    mask = np_arr > 0
    markers = label(np_arr > 1)
    distance_map = distance_transform_edt(mask)
    labels = watershed(-distance_map, markers, mask=mask)
    filled_labels = fill_labels(labels)
    return filled_labels

def postprocessing_distmap(distancemap: torch.Tensor):
    distancemap = distancemap.squeeze().detach().cpu().numpy()
    mask_fibre = distancemap >= -0.01
    mask_inreg = distancemap >= 0.9
    mask_inreg = label(binary_fill_holes(mask_inreg))
    labels = watershed(distancemap, mask_inreg, mask=mask_fibre)
    filled_labels = fill_labels(labels)
    expanded_labels = expand_labels(label_image=filled_labels, distance=1)
    return expanded_labels

def postprocessing_distmap_invedge(distancemap: torch.Tensor):
    distancemap = distancemap.squeeze().detach().cpu().numpy()
    mask_fibre = distancemap >= -0.15
    mask_inreg = np.logical_and(cv.GaussianBlur(distancemap,(1,1),0) <= 0.25, mask_fibre)
    mask_inreg = label(mask_inreg)
    mask_inreg = remove_small_objects(mask_inreg, min_size=200, connectivity=1)
    mask_inreg = label(mask_inreg)
    labels = watershed(distancemap, mask_inreg, mask=mask_fibre, compactness=10)
    filled_labels = fill_labels(labels)
    expanded_labels = expand_labels(label_image=filled_labels, distance=1)
    return expanded_labels

def postprocessing_signed_map(distancemap: torch.Tensor):
    distancemap = distancemap.squeeze().detach().cpu().numpy()
    mask_fibre = distancemap >= 1
    mask_inreg = np.logical_and(cv.GaussianBlur(distancemap,(1,1),0) >= 2.5, mask_fibre)
    mask_inreg = label(mask_inreg)
    mask_inreg = remove_small_objects(mask_inreg, min_size=200, connectivity=1)
    mask_inreg = label(mask_inreg)
    labels = watershed(distancemap, mask_inreg, mask=mask_fibre, compactness=10)
    filled_labels = fill_labels(labels)
    expanded_labels = expand_labels(label_image=filled_labels, distance=1)
    return expanded_labels

# Methods to compute distance transforms on label image
# Converts a single label into a binary mask and gets the distance transform
def dist_trans_mask(image, label_id):   
    binary_label_id = np.where(image == label_id, 1, 0) # crates binary image containing only the specified label
    dist_trans = distance_transform_edt(binary_label_id) # gets the distance transform of the binary mask
    
    return dist_trans

def erode_labels(image):
    edges = sobel(image) != 0
    image_eroded = np.where(edges == False, image, 0)
    return image_eroded

def dist_trans_labels(image):
    eroded = erode_labels(image)
    dist_map = distance_transform_edt(eroded)
    return dist_map

def dist_trans_labels_invert(image):
    distance_transform = dist_trans_labels(image)
    distance_transform = (distance_transform - np.min(distance_transform)) / (np.max(distance_transform) - np.min(distance_transform)) # normalize 0-1 range
    distance_transform = (1 - distance_transform) + 1
    distance_transform[distance_transform==2] = 0
    return distance_transform

def dist_trans_labels_invert_seed(image, sem_mask):
    distance_transform = dist_trans_labels(image)
    distance_transform = (distance_transform - np.min(distance_transform)) / (np.max(distance_transform) - np.min(distance_transform)) # normalize 0-1 range
    distance_transform = (1 - distance_transform) + 1
    distance_transform[distance_transform==2] = 0
    filled = binary_fill_holes(sem_mask > 1)
    distance_transform[filled == 1] = 1
    return distance_transform

def dist_trans_labels_seed_myelnorm(image, sem_mask):
    distance_transform = dist_trans_labels(image)
    filled = binary_fill_holes(sem_mask > 1)
    distance_transform[filled==1] = 0
    distance_transform = (distance_transform - np.min(distance_transform)) / (np.max(distance_transform) - np.min(distance_transform)) # normalize 0-1 range
    distance_transform += 1
    distance_transform[distance_transform==1] = 0
    distance_transform[filled == 1] = 2
    return distance_transform

def dist_trans_labels_bkg_minus_one(image, sem_mask):
    inner_region_mask = binary_fill_holes(sem_mask > 1)
    distance_transform = dist_trans_labels(image)
    distance_transform[inner_region_mask==1] = 0
    distance_transform = normalize(distance_transform)
    distance_transform[distance_transform==0] = -1
    distance_transform[inner_region_mask==1] = 1
    return distance_transform

def dist_trans_labels_bkg_minus_one_invedge(image, sem_mask):
    inner_region_mask = binary_fill_holes(sem_mask > 1)
    distance_transform = dist_trans_labels(image)
    distance_transform[inner_region_mask==1] = 0
    distance_transform = 1 - normalize(distance_transform)
    distance_transform[distance_transform==1] = -1
    distance_transform[inner_region_mask==1] = 0
    return distance_transform

def dist_map_signed(labels, clip_value):
    dist_trans_foreground = dist_trans_labels(labels)
    background_mask = labels == 0
    dist_trans_background = distance_transform_edt(background_mask)
    signed_distmap = dist_trans_foreground - dist_trans_background
    signed_distmap[signed_distmap > clip_value] = clip_value
    signed_distmap[signed_distmap < -clip_value] = -clip_value
    return signed_distmap

# get the distance transforms from all the masks contained in a label image
def dist_trans_labels_long(image):
    dist_trans_list = [] # creates empty list to save images of individual distance transforms
    regions = regionprops(image)

    # gets the image transform of every label and stores them as individual images
    for i in range(len(regions)):
        label_id = regions[i].label # gets the label id of the current region
        dist_trans_label = dist_trans_mask(image, label_id) # creates image containing only the specified distance transform
        dist_trans_list.append(dist_trans_label) # stores the image on the list

    # generates a new image containing all the distance transforms
    dist_trans_stack = np.stack(dist_trans_list) # creates stack from list of images (numpy arrays)
    image_dist_trans = np.max(dist_trans_stack, axis = 0) # calculates the maximum projection to get back a 2D image

    return image_dist_trans

# get a skeleton distance transforms from all the masks contained in a label image
def dist_trans_skeleton(image, skeleton, label_id):   
    binary_label_id = np.where(image == label_id, 1, 0) # crates binary image containing only the specified label
    skeleton_id = binary_label_id * skeleton # creates a binary image containing only the specified skeleton
    skeleton_inverted = 1 - skeleton_id # invert skeleton
    dist_trans = distance_transform_edt(skeleton_inverted) # gets the distance transform of the binary mask
    dist_trans = dist_trans + 0.1 # make sure skeleton is not 0
    distmap = np.where((binary_label_id == 0) == False, dist_trans, 0) # keep only the distance map for the label
    distmap = normalize(distmap) # normalize the distance map
    distmap = 1 - distmap # invert the distance map
    distmap[distmap==1] = -1 # make background -1
    return distmap

# get a skeleton distance transforms from all the masks contained in a label image
def skeletonDistanceTransform_1b1(image):
    eroded = erode_labels(image) # erode label image
    eroded = label(eroded) # relabel to have integers
    skeleton = skeletonize(eroded > 0, method='lee') # skeletonize binary image
    skeleton = skeleton > 0 # make sure it's binary (lee method returns an 8-bit image with 0 (False) and 255 (True)))
    
    dist_trans_list = [] # creates empty list to save images of individual distance transforms
    regions = regionprops(eroded)

    # gets the image transform of every label and stores them as individual images
    for i in range(len(regions)):
        label_id = regions[i].label # gets the label id of the current region
        dist_trans_label = dist_trans_skeleton(eroded, skeleton, label_id) # creates image containing only the specified distance transform
        dist_trans_list.append(dist_trans_label) # stores the image on the list

    # generates a new image containing all the distance transforms
    dist_trans_stack = np.stack(dist_trans_list) # creates stack from list of images (numpy arrays)
    image_dist_trans = np.max(dist_trans_stack, axis = 0) # calculates the maximum projection to get back a 2D image

    return image_dist_trans

def postprocessing_sdt(distancemap: torch.Tensor):
    distancemap = distancemap.squeeze().detach().cpu().numpy()
    mask_fibre = distancemap >= 0
    distancemap[distancemap < 0] = 0
    distancemap[distancemap > 1] = 1
    #distancemap = normalize(distancemap)
    #mask_inreg = np.logical_and(cv.GaussianBlur(distancemap,(1,1),0) >= 0.5, mask_fibre)
    mask_inreg = np.logical_and(distancemap >= 0.7, mask_fibre)
    mask_inreg = label(mask_inreg)
    mask_inreg = remove_small_objects(mask_inreg, min_size=700, connectivity=1)
    mask_inreg = label(mask_inreg)
    labels = watershed(-distancemap, mask_inreg, mask=mask_fibre, connectivity=1, compactness=0.5)
    filled_labels = fill_labels(labels)
    #expanded_labels = expand_labels(label_image=filled_labels, distance=1)
    return filled_labels

# Set glasbey cmap on image (returns RGB)
def set_glasbey_cmap(image):
    l=cc.cm.glasbey_bw_minc_20_minl_30_r.colors            
    l[0]=[0,0,0]
    cmap_lab = LinearSegmentedColormap.from_list('my_list', l, N=1000)
    colored_image = np.uint8(cmap_lab(image) * 255)
    return colored_image

# Get glasbey cmap
def get_glasbey_cmap():
    l=cc.cm.glasbey_bw_minc_20_minl_30_r.colors            
    l[0]=[0,0,0]
    cmap_glasbey = LinearSegmentedColormap.from_list('my_list', l, N=1000)
    return cmap_glasbey