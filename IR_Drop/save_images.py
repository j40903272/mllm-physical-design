import base64
import numpy as np
from PIL import Image
import os
import io

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def display_encoded_image(base64_image):
    # Decode the Base64 string
    image_data = base64.b64decode(base64_image)

    # Convert the decoded data to a BytesIO object
    image_bytes = io.BytesIO(image_data)

    # Open the image using PIL
    image = Image.open(image_bytes)

    # Display the image
    image.show()


def save_image(file_path, save_folder):

    image_array = np.load(file_path)
    batch_image = image_array.transpose(2,0,1)

    for i, image in enumerate(batch_image):
        image_pil = Image.fromarray(np.uint8(image * 255))
        save_path = os.path.join(save_folder, f"{i}.png")
        image_pil.save(save_path)
  
def save_grid_image(image_arrs, save_folder):
    """
    take grid images then save it
    """
    # multiple feature images
    if isinstance(image_arrs, list):
        for i, image_arr in enumerate(image_arrs):
            image_pil = Image.fromarray(np.uint8(image_arr))
            save_path = os.path.join(save_folder, f"{i}.png")
            image_pil.save(save_path)
    else: # single label image
        image_pil = Image.fromarray(np.uint8(image_arrs))
        save_path = os.path.join(save_folder, "0.png")
        image_pil.save(save_path)
