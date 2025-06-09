import numpy as np
import cv2
from scipy.stats import kurtosis

feat_pool = {
    'rudy_gradient_variability': 'the variation in gradient changes across the rudy map indicating potential areas of abrupt routing demand shifts',
    'clustered_macro_distance_std': 'the standard deviation of distances between clustered groups of macros',
    'rudy_pin_clustering_coefficient': 'a measure of how many rudy pins cluster together relative to the total number of rudy pins',
    'macro_density_gradient': 'the change in macro density across the layout, impacting local congestion',
    'macro_aspect_ratio_variance': 'the variance in aspect ratios of macros, indicating potential alignment and spacing issues that may impact congestion',
    'macro_compactness_index': 'a measure of how closely packed the macros are, potentially affecting routing paths and congestion',
    'rudy_pin_compaction_ratio': 'the ratio of compacted rudy pin clusters to the total number of rudy pins, indicating areas with high potential routing conflicts',
    'macro_variability_coefficient': 'a measure of the consistency in macro sizes and shapes relative to each other, potentially affecting congestion balance',
    'macro_symmetry_coefficient': 'a measure of the symmetry in macro placements relative to the overall layout, potentially influencing uniformity in congestion distribution',
    'macro_cluster_density_contrast': 'the contrast in density between clustered groups of macros and their surrounding layout areas, indicating potential localized congestion pressure',
    'rudy_pin_distribution_kurtosis': 'a measure of the peakedness or flatness in the distribution of rudy pins across the layout, indicating potential areas of concentrated or dispersed routing demand',
    'localized_rudy_variability_coefficient': 'a measure of the variation in RUDY intensity within localized regions, indicating potential micro-level congestion fluctuations',
    'macro_distribution_clarity_index': 'a measure of how distinct macro distributions are across the layout, indicating clarity in separation and potential influence on congestion distribution',
    'rudy_direction_consistency_index': 'a measure of the uniformity in the directional flow of RUDY intensity, indicating how consistent the routing demand is across the layout',
    'rudy_pin_area_masking_index': 'the ratio of the area masked by rudy pin regions relative to the total layout, indicating potential routing blockages',
    'rudy_pin_gradient_convergence': 'a measure of how gradients in the rudy pin map converge into specific regions, indicating high-density pin clusters',
    'rudy_intensity_symmetry_index': 'a measure of the symmetry in the RUDY intensity map across the layout, indicating uniformity in routing demand distribution',
    'rudy_deviation_effect_index': 'a measure of the deviation of RUDY intensities from the mean, indicating areas of abnormal routing demand',
    'demarcated_macro_proximity_index': 'a measure of how close macros are to predefined boundary regions, potentially affecting routing and congestion near layout edges',
    'macro_surface_irregularity_index': 'a measure of the irregularity in macro surface shapes, which can impact routing paths and layout clarity',
    'macro_rudy_boundary_interaction_index': 'a measure of the interaction between macros and high RUDY regions, indicating potential congestion hotspots',
    'pin_density_peak_contrast': 'the contrast between peak pin density regions and their surroundings, indicating areas of abrupt routing demand changes',
    'rudy_pin_density_flux_index': 'a measure of the rate of change in rudy pin density across the layout, indicating dynamic routing demand shifts',
    'high_density_rudy_ratio': 'the ratio of areas with high RUDY intensity to the total layout area, indicating overall routing demand hotspots',
    'high_density_rudy_pin_ratio': 'the ratio of areas with high RUDY pin intensity to the total layout area, indicating localized pin density hotspots'
}

def rudy_gradient_variability(images):
    tiles_size = 2.25
    macro_image = images[0]
    rudy_image = images[1]
    rudy_pin_image = images[2]
    
    image_height, image_width = macro_image.shape
    total_image_area = image_width * image_height
    
    # Convert macro image to [0, 255] range
    macro_image = np.uint8(macro_image * 255)
    _, binary_image = cv2.threshold(macro_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_macros = len(contours)

    # Compute gradients using the Sobel operator
    grad_x = cv2.Sobel(rudy_image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(rudy_image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)

    # Calculate gradient variability
    gradient_variability = np.var(magnitude)

    feature_value = gradient_variability * (tiles_size ** 2)  # Convert to um^2

    return {"rudy_gradient_variability": feature_value}


def clustered_macro_distance_std(images):
    tiles_size = 2.25
    macro_image = images[0]
    rudy_image = images[1]
    rudy_pin_image = images[2]
    
    # Convert macro image to 0-255
    macro_image = np.uint8(macro_image * 255)
    # Threshold and find contours
    _, binary_image = cv2.threshold(macro_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate centers of mass for each contour (macro)
    centers = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centers.append((cX, cY))

    # Calculate pairwise distances
    distances = []
    num_centers = len(centers)
    for i in range(num_centers):
        for j in range(i + 1, num_centers):
            dist = np.sqrt((centers[i][0] - centers[j][0]) ** 2 + (centers[i][1] - centers[j][1]) ** 2)
            distances.append(dist * tiles_size)  # Convert pixel distance to um

    # Calculate standard deviation of distances
    if distances:
        std_distance = np.std(distances)
    else:
        std_distance = 0.0
    
    return {"clustered_macro_distance_std": std_distance}

def rudy_pin_clustering_coefficient(images):
    tiles_size = 2.25
    macro_image = images[0]
    rudy_image = images[1]
    rudy_pin_image = images[2]
    
    image_height, image_width = macro_image.shape
    total_image_area = image_width * image_height
    
    macro_image = np.uint8(macro_image * 255)
    _, binary_image = cv2.threshold(macro_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_macros = len(contours)
    
    # Convert rudy_pin_image to binary
    _, rudy_pin_binary = cv2.threshold(rudy_pin_image, 0.5, 1, cv2.THRESH_BINARY)
    
    # Find contours of the rudy pins
    contours, _ = cv2.findContours(rudy_pin_binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    total_rudy_pins = len(contours)
    clustered_rudy_pins = 0
    
    # Calculate the clustering coefficient
    for contour in contours:
        if len(contour) > 1:
            clustered_rudy_pins += 1
            
    clustering_coefficient = clustered_rudy_pins / total_rudy_pins if total_rudy_pins > 0 else 0
    
    return {"rudy_pin_clustering_coefficient": clustering_coefficient}

def macro_density_gradient(images):
    tiles_size = 2.25
    macro_image = images[0]
    rudy_image = images[1]
    rudy_pin_image = images[2]
    
    image_height, image_width = macro_image.shape
    total_image_area = image_width * image_height
    
    # Convert macro image to binary [0, 255]
    macro_image = np.uint8(macro_image * 255)
    _, binary_image = cv2.threshold(macro_image, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours to get macro regions
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calculate macro density per region
    macro_density = np.zeros((image_height, image_width))
    for contour in contours:
        mask = np.zeros_like(binary_image)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        macro_density += mask

    # Gradient of the macro density
    gradient_x = cv2.Sobel(macro_density, cv2.CV_64F, 1, 0, ksize=5)
    gradient_y = cv2.Sobel(macro_density, cv2.CV_64F, 0, 1, ksize=5)
    
    # Calculate the magnitude of gradients
    gradient_magnitude = cv2.magnitude(gradient_x, gradient_y)

    # Calculate the average gradient magnitude in micrometers
    macro_density_gradient_um = np.sum(gradient_magnitude) / (image_height * image_width)
    macro_density_gradient_um *= tiles_size  # Convert to micrometers

    return {"macro_density_gradient": macro_density_gradient_um}

def macro_aspect_ratio_variance(images):
    tiles_size = 2.25
    macro_image = images[0]
    
    # Convert the macro image from [0-1] to [0-255]
    macro_image = np.uint8(macro_image * 255)
    
    # Threshold the macro image to create a binary image
    _, binary_image = cv2.threshold(macro_image, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours of the macros
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calculate aspect ratio for each macro and store them
    aspect_ratios = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / float(h)
        aspect_ratios.append(aspect_ratio)
    
    # Calculate the variance of aspect ratios
    aspect_ratio_variance = np.var(aspect_ratios) * (tiles_size ** 2)
    
    # Return the feature
    return {"macro_aspect_ratio_variance": aspect_ratio_variance}


def macro_compactness_index(images):
    tiles_size = 2.25
    macro_image = images[0]
    rudy_image = images[1]
    rudy_pin_image = images[2]

    image_height, image_width = macro_image.shape
    total_image_area = image_width * image_height * tiles_size * tiles_size

    # Convert macro image to [0-255]
    macro_image = np.uint8(macro_image * 255)

    # Threshold and find contours
    _, binary_image = cv2.threshold(macro_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate total macro area and perimeter
    total_macro_area = 0
    total_perimeter = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        total_macro_area += area
        total_perimeter += perimeter

    # Convert to real-world units (um^2 for area)
    total_macro_area_um2 = total_macro_area * (tiles_size ** 2)
    
    # Calculate compactness index
    macro_compactness_index = (total_perimeter ** 2) / total_macro_area_um2 if total_macro_area_um2 else 0

    return {"macro_compactness_index": macro_compactness_index}

def rudy_pin_compaction_ratio(images):
    tiles_size = 2.25
    macro_image = images[0]
    rudy_image = images[1]
    rudy_pin_image = images[2]
    
    # Convert macro image to [0-255] grayscale
    macro_image = np.uint8(macro_image * 255)
    
    # Calculate total number of pixels in the image
    image_height, image_width = macro_image.shape
    total_image_area = image_width * image_height

    # Determine number of macro blocks - not needed for this calculation
    _, binary_image = cv2.threshold(macro_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_macros = len(contours)

    # Process rudy pin image to find clusters
    _, binary_rudy_pin = cv2.threshold(rudy_pin_image, 0.5, 1, cv2.THRESH_BINARY)
    binary_rudy_pin = np.uint8(binary_rudy_pin * 255)

    # Find contours in the binary Rudy pin image
    contours, _ = cv2.findContours(binary_rudy_pin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calculate pin area and compacted pin area
    total_pin_area = cv2.countNonZero(binary_rudy_pin) * (tiles_size ** 2)
    num_clusters = len(contours)
    
    # Compute the compacted pin area 
    compacted_pin_area = sum(cv2.contourArea(c) for c in contours) * (tiles_size ** 2)

    # Calculate the compaction ratio
    rudy_pin_compaction_ratio = compacted_pin_area / total_pin_area if total_pin_area > 0 else 0

    return {"rudy_pin_compaction_ratio": rudy_pin_compaction_ratio}

def macro_variability_coefficient(images):
    tiles_size = 2.25
    macro_image = images[0]
    rudy_image = images[1]
    rudy_pin_image = images[2]
    
    image_height, image_width = macro_image.shape
    total_image_area = image_width * image_height
    
    macro_image = np.uint8(macro_image * 255)
    _, binary_image = cv2.threshold(macro_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calculate macro areas
    macro_areas_um2 = []
    for contour in contours:
        area = cv2.contourArea(contour) * (tiles_size**2)
        macro_areas_um2.append(area)
    
    num_macros = len(macro_areas_um2)
    
    if num_macros > 0:
        # Calculate average and standard deviation of macro areas
        mean_area = np.mean(macro_areas_um2)
        std_dev_area = np.std(macro_areas_um2)
        
        # Variability coefficient
        variability_coefficient = std_dev_area / mean_area
    else:
        variability_coefficient = 0
    
    return {"macro_variability_coefficient": variability_coefficient}

def macro_symmetry_coefficient(images):
    tiles_size = 2.25
    macro_image = images[0]
    rudy_image = images[1]
    rudy_pin_image = images[2]
    
    image_height, image_width = macro_image.shape
    total_image_area = image_width * image_height
    
    # Convert macro image to [0, 255] and threshold it
    macro_image = np.uint8(macro_image * 255)
    _, binary_image = cv2.threshold(macro_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Split the image into left and right halves
    left_half = binary_image[:, :image_width // 2]
    right_half = binary_image[:, image_width // 2:]
    
    # Calculate areas of macros on each half
    left_area = cv2.countNonZero(left_half)
    right_area = cv2.countNonZero(right_half)
    
    # Convert pixel areas to micrometers
    left_area_um = left_area * tiles_size * tiles_size
    right_area_um = right_area * tiles_size * tiles_size
    
    # Compute symmetry coefficient
    symmetry_coefficient = abs(left_area_um - right_area_um) / (left_area_um + right_area_um)
    
    return {"macro_symmetry_coefficient": symmetry_coefficient}

def macro_cluster_density_contrast(images):
    tiles_size = 2.25
    macro_image = images[0]
    rudy_image = images[1]
    rudy_pin_image = images[2]
    
    image_height, image_width = macro_image.shape
    total_image_area = image_width * image_height
    
    macro_image = np.uint8(macro_image * 255)
    _, binary_image = cv2.threshold(macro_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_macros = len(contours)
    
    # Calculate macro coverage area
    macro_area = cv2.countNonZero(binary_image)
    macro_area_um = macro_area * tiles_size ** 2
    
    # Calculate average RUDY in macro areas
    rudy_macro_values = rudy_image[binary_image > 0]
    average_rudy_macro = np.mean(rudy_macro_values)
    
    # Calculate overall average RUDY
    average_rudy_total = np.mean(rudy_image)
    
    # Calculate density contrast
    density_contrast = (average_rudy_macro - average_rudy_total) / average_rudy_total
    
    feature_value = density_contrast
    
    return {"macro_cluster_density_contrast": feature_value}

def rudy_pin_distribution_kurtosis(images):
    tiles_size = 2.25
    macro_image = images[0]
    rudy_image = images[1]
    rudy_pin_image = images[2]
    
    image_height, image_width = macro_image.shape
    total_image_area = image_width * image_height

    # Scale macro_image to [0, 255]
    macro_image = np.uint8(macro_image * 255)
    
    # Threshold macro image to find contours
    _, binary_image = cv2.threshold(macro_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_macros = len(contours)

    # Calculate RUDY pin distribution kurtosis
    rudy_pin_flat = rudy_pin_image.flatten()
    rudy_pin_distribution_kurtosis_value = kurtosis(rudy_pin_flat)

    return {"rudy_pin_distribution_kurtosis": rudy_pin_distribution_kurtosis_value}

def localized_rudy_variability_coefficient(images):
    tiles_size = 2.25
    macro_image = images[0]
    rudy_image = images[1]
    rudy_pin_image = images[2]
    
    image_height, image_width = macro_image.shape
    total_image_area = image_width * image_height
    
    # Scale macro image to [0, 255]
    macro_image = np.uint8(macro_image * 255)
    
    _, binary_image = cv2.threshold(macro_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_macros = len(contours)
    
    # Tile size in pixels
    tile_size_px = int(256 / 16)
    
    rudy_variability = []
    
    # Iterate over tiles
    for y in range(0, image_height, tile_size_px):
        for x in range(0, image_width, tile_size_px):
            # Extract the tile from the RUDY image
            tile = rudy_image[y:y + tile_size_px, x:x + tile_size_px]
            
            if tile.size > 0:
                # Flatten the tile to compute statistics
                tile_values = tile.flatten()
                # Calculate mean and standard deviation
                mean_intensity = np.mean(tile_values)
                std_dev_intensity = np.std(tile_values)
                
                if mean_intensity != 0:
                    # Coefficient of Variation (CV) = std_dev / mean
                    cv = std_dev_intensity / mean_intensity
                else:
                    cv = 0
                
                rudy_variability.append(cv)
    
    # Calculate the overall variability coefficient as the mean of CVs
    if rudy_variability:
        overall_variability_coefficient = np.mean(rudy_variability)
    else:
        overall_variability_coefficient = 0

    return {"localized_rudy_variability_coefficient": overall_variability_coefficient}


def macro_distribution_clarity_index(images):
    tiles_size = 2.25
    macro_image = images[0]
    rudy_image = images[1]
    rudy_pin_image = images[2]
    
    # Convert macro image to [0-255]
    macro_image = np.uint8(macro_image * 255)

    # Calculate total image area in um^2
    image_height, image_width = macro_image.shape
    total_image_area_um2 = (image_width * tiles_size) * (image_height * tiles_size)

    # Threshold the macro image to create a binary image
    _, binary_image = cv2.threshold(macro_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calculate macro area
    macro_area_pixels = sum(cv2.contourArea(contour) for contour in contours)
    macro_area_um2 = macro_area_pixels * (tiles_size ** 2)

    # Calculate the bounding area of the RUDY areas (define contrast)
    rudy_contrast = cv2.Laplacian(rudy_image, cv2.CV_64F).var()
    
    # Combine contrast info with macro area clarity
    clarity_index = (rudy_contrast + len(contours)) / (macro_area_um2 / total_image_area_um2)
    
    return {"macro_distribution_clarity_index": clarity_index}

def rudy_direction_consistency_index(images):
    tiles_size = 2.25
    macro_image = images[0]
    rudy_image = images[1]
    rudy_pin_image = images[2]
    
    # Convert macro image to uint8
    macro_image = np.uint8(macro_image * 255)
    _, binary_image = cv2.threshold(macro_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_macros = len(contours)
    
    # Gradient calculation for RUDY image
    grad_x = cv2.Sobel(rudy_image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(rudy_image, cv2.CV_64F, 0, 1, ksize=3)

    # Normalize gradients
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    angle = np.arctan2(grad_y, grad_x)

    # Filter out zero magnitudes to avoid division by zero
    valid_magnitudes = magnitude > 0
    angle_filtered = angle[valid_magnitudes]

    # Direction consistency calculation
    # Calculate the variance of angles (directional consistency)
    if angle_filtered.size > 0:
        consistency_index = 1 - (np.var(angle_filtered) / (2 * np.pi))
    else:
        consistency_index = 0

    return {"rudy_direction_consistency_index": consistency_index}

def rudy_pin_area_masking_index(images):
    tiles_size = 2.25
    macro_image = images[0]
    rudy_image = images[1]
    rudy_pin_image = images[2]
    
    image_height, image_width = macro_image.shape
    total_image_area = image_width * image_height
    
    # Convert macro image to [0, 255] and create a binary mask
    macro_image_uint8 = np.uint8(macro_image * 255)
    _, macro_mask = cv2.threshold(macro_image_uint8, 127, 255, cv2.THRESH_BINARY)
    
    # Calculate the area of the RUDY pin image that is masked by macros
    rudy_pin_area = np.sum(rudy_pin_image)
    masked_rudy_pin_area = np.sum(rudy_pin_image * (macro_mask / 255))
    
    # Calculate the rudy pin area masking index
    if rudy_pin_area == 0:
        rudy_pin_area_masking_index = 0
    else:
        rudy_pin_area_masking_index = masked_rudy_pin_area / rudy_pin_area
    
    # Convert pixels to area in um^2
    total_image_area_um2 = total_image_area * (tiles_size ** 2)
    rudy_pin_area_masked_um2 = rudy_pin_area_masking_index * total_image_area_um2
    
    return {"rudy_pin_area_masking_index": rudy_pin_area_masking_index}

def rudy_pin_gradient_convergence(images):
    tiles_size = 2.25
    macro_image = images[0]
    rudy_image = images[1]
    rudy_pin_image = images[2]

    # Convert macro image to [0-255]
    macro_image = np.uint8(macro_image * 255)

    # Convert macro image to binary
    _, binary_image = cv2.threshold(macro_image, 127, 255, cv2.THRESH_BINARY)

    # Find contours in the macro image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_macros = len(contours)

    # Compute gradients of the RUDY pin image
    grad_x = cv2.Sobel(rudy_pin_image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(rudy_pin_image, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate the gradient magnitude and direction
    magnitude = cv2.magnitude(grad_x, grad_y)
    angle = cv2.phase(grad_x, grad_y)

    # Normalize the magnitude
    magnitude /= magnitude.max()

    # Calculate the histogram of gradient directions
    hist_bins = 36
    hist_range = (0, 2 * np.pi)
    hist, _ = np.histogram(angle, bins=hist_bins, range=hist_range, weights=magnitude)

    # Normalize the histogram
    hist /= hist.sum()

    # Compute the convergence feature as the entropy of the distribution
    epsilon = 1e-5  # small value to avoid log(0)
    entropy = -np.sum(hist * np.log(hist + epsilon))

    # Adjust the entropy to fit as a convergence metric
    feature_value = 1.0 / (entropy + epsilon)

    return {"rudy_pin_gradient_convergence": feature_value}


def rudy_intensity_symmetry_index(images):
    tiles_size = 2.25
    macro_image = images[0]
    rudy_image = images[1]
    rudy_pin_image = images[2]
    
    image_height, image_width = macro_image.shape
    total_image_area = image_width * image_height
    
    macro_image = np.uint8(macro_image * 255)
    _, binary_image = cv2.threshold(macro_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_macros = len(contours)
    
    # Calculate symmetry index for rudy_image
    left_half = rudy_image[:, :image_width // 2]
    right_half = rudy_image[:, image_width // 2:]
    
    # Flip right half horizontally
    flipped_right_half = cv2.flip(right_half, 1)

    # Compute the absolute difference between left half and flipped right half
    diff_image = cv2.absdiff(left_half, flipped_right_half)

    # Sum of differences as a measure of asymmetry
    asymmetry_measure = np.sum(diff_image)

    # Normalize the measure relative to the total possible maximal intensity difference
    max_intensity = 1.0  # Since rudy_image is in range [0, 1]
    max_possible_diff = max_intensity * (image_width // 2) * image_height
    symmetry_index = 1 - (asymmetry_measure / max_possible_diff)
    
    return {"rudy_intensity_symmetry_index": symmetry_index}

def rudy_deviation_effect_index(images):
    tiles_size = 2.25
    macro_image = images[0]
    rudy_image = images[1]
    rudy_pin_image = images[2]
    
    image_height, image_width = macro_image.shape
    total_image_area = image_width * image_height
    
    # Convert macro image from [0, 1] to [0, 255]
    macro_image = np.uint8(macro_image * 255)
    _, binary_image = cv2.threshold(macro_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_macros = len(contours)
    
    # Calculate macro area in pixel units
    macro_area_pixel = np.sum(binary_image == 255)
    
    # Calculate macro area in um^2
    macro_area_um2 = macro_area_pixel * (tiles_size ** 2)
    
    # Calculate average RUDY intensity
    avg_rudy_intensity = np.mean(rudy_image)
    
    # Calculate deviation of RUDY intensity
    deviation_rudy_intensity = np.std(rudy_image)
    
    # Calculate a measure of how deviations in RUDY intensity correlate with macro density
    # A simplified approach assuming positive correlation
    correlation_measure = deviation_rudy_intensity * macro_area_um2 / total_image_area
    
    feature_value = correlation_measure
    
    return {"rudy_deviation_effect_index": feature_value}

def demarcated_macro_proximity_index(images):
    tiles_size = 2.25
    macro_image = images[0]
    rudy_image = images[1]
    
    image_height, image_width = macro_image.shape
    
    # Convert macro image to [0-255]
    macro_image = np.uint8(macro_image * 255)
    
    # Threshold the macro image to find contours of the macros
    _, binary_image = cv2.threshold(macro_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_macros = len(contours)
    
    # Threshold for defining high-density areas in RUDY map
    rudy_threshold = 0.5
    
    # Identifying high-density zones in RUDY
    high_density_zone = rudy_image > rudy_threshold
    
    # Initialize a variable to accumulate proximity measure
    proximity_sum = 0
    
    # Calculate proximity index
    for contour in contours:
        macro_mask = np.zeros_like(macro_image)
        cv2.drawContours(macro_mask, [contour], -1, 255, thickness=cv2.FILLED)
        
        intersection = np.logical_and(macro_mask > 0, high_density_zone)
        intersection_area = np.sum(intersection) * tiles_size * tiles_size
        
        # Proximity for current macro
        proximity_sum += intersection_area
        
    # Normalizing by total macro area
    total_macro_area = np.sum(macro_image > 0) * tiles_size * tiles_size
    demarcated_macro_proximity_index = (proximity_sum / (total_macro_area + 1e-5)) if total_macro_area > 0 else 0
    
    return {"demarcated_macro_proximity_index": demarcated_macro_proximity_index}

def macro_surface_irregularity_index(images):
    import cv2
    import numpy as np
    
    tiles_size = 2.25
    macro_image = images[0]
    rudy_image = images[1]
    rudy_pin_image = images[2]
    
    image_height, image_width = macro_image.shape
    total_image_area = image_width * image_height * (tiles_size ** 2)  # Convert area to micrometers

    macro_image = np.uint8(macro_image * 255)  # Convert macro image to [0-255]
    _, binary_image = cv2.threshold(macro_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_macros = len(contours)

    # Calculate total perimeter and area of macros
    total_perimeter = 0
    total_macro_area = 0
    for contour in contours:
        total_perimeter += cv2.arcLength(contour, True)
        total_macro_area += cv2.contourArea(contour)

    # Convert perimeter and area to micrometers
    total_perimeter_um = total_perimeter * tiles_size
    total_macro_area_um = total_macro_area * (tiles_size ** 2)

    # Irregularity index calculation
    if total_macro_area_um > 0:
        irregularity_index = total_perimeter_um / total_macro_area_um
    else:
        irregularity_index = 0

    return {"macro_surface_irregularity_index": irregularity_index}

def macro_rudy_boundary_interaction_index(images):
    tiles_size = 2.25
    macro_image = images[0]
    rudy_image = images[1]
    rudy_pin_image = images[2]
    
    image_height, image_width = macro_image.shape
    total_image_area = image_width * image_height
    
    # Convert macro image to [0-255]
    macro_image = np.uint8(macro_image * 255)
    _, binary_image = cv2.threshold(macro_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_macros = len(contours)
    
    # Calculate interaction index
    interaction_index = 0
    
    # Iterate over each contour
    for contour in contours:
        # Create a mask of the macro contour
        contour_mask = np.zeros((image_height, image_width), dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, (1), thickness=cv2.FILLED)
        
        # Calculate the intersection of the macro contour with RUDY pin image
        interaction_area = np.sum(contour_mask * rudy_pin_image)
        
        # Convert pixels to area in um^2 (each pixel is 2.25um x 2.25um)
        interaction_area_um = interaction_area * (tiles_size ** 2)
        
        # Check if the averaged RUDY pin density along macro boundary indicates congestion
        if interaction_area > 0:
            interaction_index += interaction_area_um / cv2.arcLength(contour, True) * tiles_size

    return {"macro_rudy_boundary_interaction_index": interaction_index}

def pin_density_peak_contrast(images):
    tiles_size = 2.25
    macro_image = images[0]
    rudy_image = images[1]
    rudy_pin_image = images[2]

    image_height, image_width = macro_image.shape
    total_image_area = image_width * image_height

    # Convert macro image to [0-255]
    macro_image = np.uint8(macro_image * 255)
    _, binary_image = cv2.threshold(macro_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_macros = len(contours)

    # Compute pin density peaks
    # Using a kernel to emphasize local density peaks
    kernel_size = 3  # Define an appropriate kernel size
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
    smoothed_rudy_pin = cv2.filter2D(rudy_pin_image, -1, kernel)
    
    # Find pin density peak contrast
    pin_density_peak = np.max(smoothed_rudy_pin)
    avg_density_around_peaks = np.mean(smoothed_rudy_pin)
    
    # Compute the contrast
    contrast = pin_density_peak - avg_density_around_peaks

    # Convert image dimensions to a physical measurement
    contrast_um = contrast * tiles_size  # Convert to um using the tile size
    
    return {"pin_density_peak_contrast": contrast_um}

def rudy_pin_density_flux_index(images):
    tiles_size = 2.25
    macro_image = images[0]
    rudy_image = images[1]
    rudy_pin_image = images[2]
    
    image_height, image_width = macro_image.shape
    total_image_area = image_width * image_height
    
    # Convert macro image to [0, 255]
    macro_image = np.uint8(macro_image * 255)
    _, binary_image = cv2.threshold(macro_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_macros = len(contours)
    
    # Process the RUDY pin image
    # Compute the gradients along the x and y axis
    grad_x = cv2.Sobel(rudy_pin_image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(rudy_pin_image, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute the gradient magnitude
    grad_magnitude = cv2.magnitude(grad_x, grad_y)
    
    # Calculate the mean of the gradient magnitude to represent density flux
    mean_flux = np.mean(grad_magnitude)
    
    # Normalize to area in square micrometers (um^2)
    pixel_area = tiles_size ** 2
    total_area_um2 = total_image_area * pixel_area
    
    feature_value = mean_flux / total_area_um2
    
    return {"rudy_pin_density_flux_index": feature_value}

def high_density_rudy_ratio(images):
    image = images[1]
    total_area = image.shape[0] * image.shape[1]
    mean_rudy = np.mean(image)
    high_density_rudy_ratio = (image > mean_rudy).sum() /  total_area
    
    return {
        "high_density_rudy_ratio": high_density_rudy_ratio,
    }
    
def high_density_rudy_pin_ratio(images):
    image = images[2]
    total_area = image.shape[0] * image.shape[1]
    mean_rudy_pin = np.mean(image)
    high_density_rudy_pin_ratio = (image > mean_rudy_pin).sum() /  total_area
    
    return {
        "high_density_rudy_pin_ratio": high_density_rudy_pin_ratio,
    }

feat_func_list = [
 rudy_gradient_variability,
 clustered_macro_distance_std,
 rudy_pin_clustering_coefficient,
 macro_density_gradient,
 macro_aspect_ratio_variance,
 macro_compactness_index,
 rudy_pin_compaction_ratio,
 macro_variability_coefficient,
 macro_symmetry_coefficient,
 macro_cluster_density_contrast,
 rudy_pin_distribution_kurtosis,
 localized_rudy_variability_coefficient,
 macro_distribution_clarity_index,
 rudy_direction_consistency_index,
 rudy_pin_area_masking_index,
 rudy_pin_gradient_convergence,
 rudy_intensity_symmetry_index,
 rudy_deviation_effect_index,
 demarcated_macro_proximity_index,
 macro_surface_irregularity_index,
 macro_rudy_boundary_interaction_index,
 pin_density_peak_contrast,
 rudy_pin_density_flux_index,
 high_density_rudy_ratio,
 high_density_rudy_pin_ratio,
]