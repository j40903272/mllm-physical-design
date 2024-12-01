import os
import numpy as np
from PIL import Image
import re
from grid import RGB_grid

from save_images import save_image, save_grid_image

def get_sorted_files_by_design(feature_path, label_path):
    """
    Takes feature_path and label_path, and returns two lists of file paths,
    sorted by design type.

    Parameters:
    - feature_path: Path to the feature files.
    - label_path: Path to the label files.

    Returns:
    - feature_files_sorted: List of feature file paths, sorted by design type.
    - label_files_sorted: List of label file paths, sorted by design type.
    """
    # List of design types you provided
    design_types = [
        'RISCY-a',
        'RISCY-FPU-a',
        'zero-riscy-a',
        'RISCY-b',
        'RISCY-FPU-b',
        'zero-riscy-b'
    ]
    
    # Compile regex patterns to match design types in filenames
    design_patterns = {design: re.compile(re.escape(design), re.IGNORECASE) for design in design_types}

    # Function to extract design type from filename
    def extract_design_type(filename):
        for design, pattern in design_patterns.items():
            if pattern.search(filename):
                return design
        return None  # Design type not found

    # Get list of feature and label files
    feature_files = os.listdir(feature_path)
    label_files = os.listdir(label_path)

    # Create dictionaries mapping design types to files
    feature_files_by_design = {design: [] for design in design_types}
    label_files_by_design = {design: [] for design in design_types}

    # Group feature files by design type
    for f in feature_files:
        design_type = extract_design_type(f)
        if design_type:
            feature_files_by_design[design_type].append(os.path.join(feature_path, f))
        else:
            print(f"Warning: Design type not found in feature file '{f}'")

    # Group label files by design type
    for f in label_files:
        design_type = extract_design_type(f)
        if design_type:
            label_files_by_design[design_type].append(os.path.join(label_path, f))
        else:
            print(f"Warning: Design type not found in label file '{f}'")

    # Prepare sorted lists
    feature_files_sorted = []
    label_files_sorted = []

    for design in design_types:
        # Sort files within each design type if needed
        feature_files_sorted.extend(sorted(feature_files_by_design[design]))
        label_files_sorted.extend(sorted(label_files_by_design[design]))

    return feature_files_sorted, label_files_sorted


def save_sample_images(feature_path, label_path, pairwise_path, num_samples=2):
    """
    Processes feature and label images from given .npy files, optionally applies grids, and saves them to specified directories.

    Parameters:
    - feature_path: Path to the feature .npy file.
    - label_path: Path to the label .npy file.
    - with_grid: If True, apply grid overlay to images before saving.
    - num_samples: Number of samples to process (default is 10).

    The goal is to have 'input_i' with feature and label folders following the same conventions.
    """

    sample_arrs = []

    # sort the files by designs
    feature_files_sorted, label_files_sorted = get_sorted_files_by_design(feature_path, label_path)

    # loop through the first num_sample
    for i, (feature_data_path, label_data_path) in enumerate(zip(feature_files_sorted[:num_samples], label_files_sorted[:num_samples])):

        # store each sample
        sample_dict = {} 

        # paths
        sample_path = os.path.join(pairwise_path, f"sample_{i}")
        save_feature_path = os.path.join(sample_path, "feature")
        save_label_path = os.path.join(sample_path, "label")
        grid_save_feature_path = os.path.join(sample_path, "feature_grid")
        grid_save_label_path = os.path.join(sample_path, "label_grid")

        # create folders
        os.makedirs(sample_path, exist_ok=True)
        os.makedirs(save_feature_path, exist_ok=True)
        os.makedirs(save_label_path, exist_ok=True)
        os.makedirs(grid_save_feature_path, exist_ok=True)
        os.makedirs(grid_save_label_path, exist_ok=True)

        # load image arrays
        label_arr = np.load(label_data_path)
        feature_arrs = np.load(feature_data_path)

        # apply grids
        feature_arrs = feature_arrs.transpose(2, 0, 1) # transpose arrays to apply grid
        label_arr = label_arr.transpose(2, 0, 1)

        # store grids and masks
        feature_arrs_grid = []
        feature_masks = [] # masks are dict with 3 levels
        for i, feature_arr in enumerate(feature_arrs):
            feature_grid, masks = RGB_grid(feature_arr) # masks is for a single image
            feature_arrs_grid.append(feature_grid)
            feature_masks.append(masks)

        label_arr_grid, label_masks = RGB_grid(label_arr[0])

        # save images
        save_image(feature_data_path, save_feature_path)
        save_image(label_data_path, save_label_path)
        save_grid_image(feature_arrs_grid, grid_save_feature_path)
        save_grid_image(label_arr_grid, grid_save_label_path)

        # store into list
        sample_dict['feature_arrs'] = feature_arrs
        sample_dict['label_arr'] = label_arr
        sample_dict['feature_arrs_grid'] = feature_arrs_grid
        sample_dict['label_arr_grid'] = label_arr_grid
        sample_dict['feature_masks'] = feature_masks
        sample_dict['label_masks'] = label_masks

        sample_arrs.append(sample_dict)

    return sample_arrs
        