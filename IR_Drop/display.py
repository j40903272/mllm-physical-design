from PIL import Image
import numpy as np

def display_array(image_arr):
    """
    Displays an array as an image using PIL. Handles array shape, list input,
    normalization, and conversion to uint8.

    Parameters:
    - image_arr: Input array or list of arrays to display as an image.

    Returns:
    - None
    """
    
    # Handle list input (use the first array in the list)
    if isinstance(image_arr, list):
        image_arr = image_arr[0]

    # Squeeze any singleton dimensions
    image_arr = np.squeeze(image_arr)

    # Handle different possible shapes
    if len(image_arr.shape) == 2:
        # Already 2D, proceed
        pass
    elif len(image_arr.shape) == 3:
        # Check for shapes like (256, 256, 1) or (1, 256, 256)
        if image_arr.shape[2] == 3 or image_arr.shape[2] == 4:
            # Assume it's a color image (RGB or RGBA), proceed
            pass
        elif image_arr.shape[2] == 1:
            image_arr = image_arr[:, :, 0]
        elif image_arr.shape[0] == 1:
            image_arr = image_arr[0, :, :]
        else:
            print(f"Error: Expected a 2D array after adjustments, got shape {image_arr.shape}")
            return
    else:
        print(f"Error: Expected a 2D array, got shape {image_arr.shape}")
        return

    # Confirm the array is now 2D or 3D with 3 channels (RGB)
    if len(image_arr.shape) == 2:
        mode = 'L'  # Grayscale
    elif len(image_arr.shape) == 3 and image_arr.shape[2] == 3:
        mode = 'RGB'  # Color image
    else:
        print(f"Error: Unsupported array shape {image_arr.shape}")
        return

    # Normalize array to 0-255 range if necessary
    if image_arr.dtype != np.uint8:
        min_val = np.min(image_arr)
        max_val = np.max(image_arr)
        if max_val == min_val:
            # Avoid division by zero; create an array of zeros
            normalized_arr = np.zeros_like(image_arr, dtype=np.uint8)
        else:
            # Shift and scale the array to 0-255
            normalized_arr = (image_arr - min_val) / (max_val - min_val) * 255
            normalized_arr = normalized_arr.astype(np.uint8)
    else:
        normalized_arr = image_arr

    # Convert array to PIL Image and display
    try:
        image_pil = Image.fromarray(normalized_arr, mode=mode)
        image_pil.show()
    except Exception as e:
        print(f"Error displaying image: {e}")
