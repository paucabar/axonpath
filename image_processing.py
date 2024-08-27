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

# Methods to fill holes on label image
# Converts a single label into a binary mask and fills its holes
def fill_mask(image, label_id, min_area):   
    binary_label_id = np.where(image == label_id, 1, 0) # crates binary image containing only the specified label
    filled = remove_small_holes(binary_label_id, min_area) # fills the binary mask
    filled_label_id = np.where(filled == 1, label_id, 0) # creates label image containing only the specified label after filling

    return filled_label_id

# Fill the holes on all the masks contained in a label image
def fill_labels(image):
    regions = regionprops(image.astype(int))

    if len(regions) > 0:
        # Initialize the result image as a copy of the input image
        image_filled = np.copy(image)

        # Fill each label one at a time and update the result image
        for i in range(len(regions)):
            label_id = regions[i].label
            filled_label = fill_mask(image, label_id, regions[i].area)
            
            # Update the result image in-place
            np.maximum(image_filled, filled_label, out=image_filled)

        return image_filled
    else:
        return image


# last layers for semantic segmentation
def last_layer_fn(pred: torch.Tensor):
    m = torch.nn.Softmax(dim=None)
    smax = m(pred)
    argmax = torch.argmax(smax, dim=1)
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
    distancemap = normalize(distancemap)
    #mask_inreg = np.logical_and(cv.GaussianBlur(distancemap,(1,1),0) >= 0.5, mask_fibre)
    mask_inreg = np.logical_and(distancemap >= 0.7, mask_fibre)
    mask_inreg = label(mask_inreg)
    mask_inreg = remove_small_objects(mask_inreg, min_size=100, connectivity=1)
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