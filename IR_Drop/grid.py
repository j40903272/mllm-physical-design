import numpy as np
from skimage.filters import threshold_multiotsu
from PIL import Image

def RGB_grid(image_array, cell_size=10, num_levels=3):
    """
    Applies a grid to the image and colors cells based on multilevel Otsu thresholds.

    Parameters:
    - image_array: Input grayscale image as a 2D numpy array.
    - cell_size: Size of the grid cells in pixels (default is 10).
    - num_levels: Number of severity levels/classes (default is 3).

    Returns:
    - rgb_image: Output RGB image with grid and cell coloring.
    - masks: A dictionary containing binary masks for each severity level.
    """
    # Ensure image_array is a 2D array
    if len(image_array.shape) == 3 and image_array.shape[2] == 1:
        image_array = image_array[:, :, 0]
    
    # Normalize the image to 0-255 range if necessary
    if np.max(image_array) <= 1.0:
        image_array = (image_array * 255).astype(np.uint8)
    else:
        image_array = image_array.astype(np.uint8)
  
    # Check if the image has enough unique values for multi-level thresholding
    unique_values = np.unique(image_array)
    if len(unique_values) < num_levels:
        # the image is black
        thresholds = [np.max(image_array) + 1] * (num_levels - 1)
    else:
        # Compute multilevel Otsu thresholds
        thresholds = threshold_multiotsu(image_array, classes=num_levels)

    # Calculate the masks
    thresholds_full = np.concatenate(([-np.inf], thresholds, [np.inf]))
    severity_levels = ['low', 'medium', 'high']
    masks = {}

    for idx, level in enumerate(severity_levels):
        mask = (image_array > thresholds_full[idx]) & (image_array <= thresholds_full[idx + 1])
        masks[level] = mask

    # Define colors for each class
    colors = [
        [0, 0, 255],    # Blue for the lowest class
        [0, 255, 0],    # Green for the middle class
        [255, 0, 0],    # Red for the highest class
    ]
    
    # Create an empty RGB image
    rgb_image = np.zeros((image_array.shape[0], image_array.shape[1], 3), dtype=np.uint8)
    
    # Draw grid and color cells based on thresholds
    for i in range(0, image_array.shape[0], cell_size):
        for j in range(0, image_array.shape[1], cell_size):
            # Ensure the cell does not exceed image dimensions
            cell_i_end = min(i + cell_size, image_array.shape[0])
            cell_j_end = min(j + cell_size, image_array.shape[1])
            
            # Extract the current cell
            cell = image_array[i:cell_i_end, j:cell_j_end]
            avg_value = np.mean(cell)
            
            # Determine the class index based on thresholds
            class_idx = np.digitize(avg_value, bins=thresholds)
            color = colors[class_idx]
            
            rgb_image[i:cell_i_end, j:cell_j_end, :] = color
            
            # Draw grid lines (white borders)
            if i + cell_size <= image_array.shape[0]:
                rgb_image[cell_i_end-1, j:cell_j_end, :] = [255, 255, 255]  # Bottom border
            if j + cell_size <= image_array.shape[1]:
                rgb_image[i:cell_i_end, cell_j_end-1, :] = [255, 255, 255]  # Right border
    
    return rgb_image, masks
