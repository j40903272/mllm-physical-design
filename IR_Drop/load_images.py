import os
from save_images import encode_image

def load_images(base_dir):
    """ 
    input: path to the pairwise folder
    output: lists

    return lists with dictionaries(samples), keys = features, labels, features_grid, label_grid
    each image is encoded into base64 images and ready to be fed into VLM
    """

    # Helper function to process an image directory
    def process_image_directory(base_path, directory_name, sort_numerically=False):
        dir_path = os.path.join(base_path, directory_name)
        images = []
        if os.path.exists(dir_path):
            # Get list of .png files
            png_files = [f for f in os.listdir(dir_path) if f.lower().endswith('.png')]
            if sort_numerically:
                # Sort the filenames numerically
                def extract_number(filename):
                    basename = os.path.splitext(filename)[0]
                    try:
                        return int(basename)
                    except ValueError:
                        return float('inf')  # Handle non-numeric filenames
                png_files_sorted = sorted(png_files, key=extract_number)
            else:
                png_files_sorted = sorted(png_files)
            for filename in png_files_sorted:
                file_path = os.path.join(dir_path, filename)
                encoded_image = encode_image(file_path)
                images.append(encoded_image)
        return images

    # Initialize list to store encoded images
    samples = []
    # Get a sorted list of all sample directories
    sample_dirs = sorted([d for d in os.listdir(base_dir) if d.startswith('sample_')])

    for sample_dir in sample_dirs:
        sample_path = os.path.join(base_dir, sample_dir)
        
        # Initialize dictionary to hold data for this sample
        sample_data = {}
        
        # Process each category using the helper function
        sample_data['feature_base64s'] = process_image_directory(sample_path, 'feature', sort_numerically=True)
        sample_data['label_base64'] = process_image_directory(sample_path, 'label')
        sample_data['feature_grid_base64s'] = process_image_directory(sample_path, 'feature_grid', sort_numerically=True)
        sample_data['label_grid_base64'] = process_image_directory(sample_path, 'label_grid')
        # Append the collected data for this sample
        samples.append(sample_data)
    
    return samples
