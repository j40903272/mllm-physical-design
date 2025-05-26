import numpy as np
import cv2

feat_pool = ["macro_spacing_std",
 "macro_boundary_distance_var",
 "pin_clustering_factor",
 "macro_diagonal_connectivity",
 "rudy_gradation_smoothness",
 "macro_edge_proximity_to_pins",
 "macro_cluster_compactness",
 "pin_density_variance",
 "pin_neighborhood_uniformity",
 "rudy_consistency_index",
 "pin_to_macro_rudy_gradient_proximity",
 "sector_rudy_disparity",
 "macro_corner_count",
 "pin_to_macro_edge_proximity_std",
 "macro_linear_alignment",
 "rudy_peak_clustering",
 "macro_pin_alignment_score",
 "pin_density_gradient",
 "macro_to_pin_cluster_proximity"]

def macro_spacing_std(images):
    tiles_size = 2.25
    macro_image = images[0]
    rudy_image = images[1]
    rudy_pin_image = images[2]
    
    image_height, image_width = macro_image.shape
    total_image_area = image_width * image_height
    
    # Convert macro_image to [0, 255]
    macro_image = np.uint8(macro_image * 255)
    _, binary_image = cv2.threshold(macro_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Extract the bounding boxes of the macros
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
    
    # Calculate the center of each macro
    macro_centers = [(x + w / 2, y + h / 2) for (x, y, w, h) in bounding_boxes]
    
    # Function to calculate Euclidean distance
    def euclidean_distance(pt1, pt2):
        return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
    
    # Calculate all pairwise distances between macro centers
    distances = []
    for i in range(len(macro_centers)):
        for j in range(i + 1, len(macro_centers)):
            dist = euclidean_distance(macro_centers[i], macro_centers[j])
            distances.append(dist * tiles_size)  # Convert to micrometers
    
    # Compute the standard deviation of the distances
    if len(distances) > 1:
        spacing_std = np.std(distances)
    else:
        spacing_std = 0.0  # No meaningful spacing if less than two macros
    
    return {"macro_spacing_std": spacing_std}

def macro_boundary_distance_var(images):
    tiles_size = 2.25  # size in micrometers
    macro_image = images[0]

    # Convert the macro image to uint8
    macro_image = np.uint8(macro_image * 255)

    # Threshold the image to get binary
    _, binary_image = cv2.threshold(macro_image, 127, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate distances from macro contours to image boundaries
    distances = []
    for contour in contours:
        # Calculate bounding rect for each macro
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate distances to each boundary (top, bottom, left, right)
        top_dist = y
        bottom_dist = macro_image.shape[0] - (y + h)
        left_dist = x
        right_dist = macro_image.shape[1] - (x + w)
        
        # Convert pixel distances to micrometers
        top_dist_um = top_dist * tiles_size
        bottom_dist_um = bottom_dist * tiles_size
        left_dist_um = left_dist * tiles_size
        right_dist_um = right_dist * tiles_size
        
        # Collect all boundary distances for the current macro
        distances.extend([top_dist_um, bottom_dist_um, left_dist_um, right_dist_um])

    # Calculate the variance of the distances
    feature_value = np.var(distances)

    return {"macro_boundary_distance_var": feature_value}


def pin_clustering_factor(images):
    tiles_size = 2.25
    macro_image = images[0]
    rudy_image = images[1]
    rudy_pin_image = images[2]
    
    image_height, image_width = macro_image.shape
    total_image_area = image_width * image_height

    # Convert macro image to [0-255] range
    macro_image = np.uint8(macro_image * 255)
    
    # Thresholding to extract macros
    _, binary_image = cv2.threshold(macro_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_macros = len(contours)
    
    # Process the rudy_pin_image
    # Convert to binary (assuming pins are prominent in the image)
    _, pin_binary = cv2.threshold(rudy_pin_image, 0.5, 1, cv2.THRESH_BINARY)
    pin_binary = np.uint8(pin_binary * 255)
    
    # Find pin contours to detect clusters
    pin_contours, _ = cv2.findContours(pin_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate pin clustering factor
    # Example: Using average area of pin clusters
    cluster_areas = [cv2.contourArea(c) for c in pin_contours]
    if cluster_areas:
        avg_cluster_area = np.mean(cluster_areas)
    else:
        avg_cluster_area = 0

    # Convert area from pixel^2 to um^2
    avg_cluster_area_um = avg_cluster_area * (tiles_size ** 2)

    # Define pin_clustering_factor based on average cluster area
    # Here a simple proportional relation; adjust based on specific criteria
    pin_clustering_factor = avg_cluster_area_um / total_image_area

    return {"pin_clustering_factor": pin_clustering_factor}


def macro_diagonal_connectivity(images):
    tiles_size = 2.25
    macro_image = images[0]
    rudy_image = images[1]
    rudy_pin_image = images[2]
    
    image_height, image_width = macro_image.shape
    total_image_area = image_width * image_height
    
    # Convert macro image to [0-255] scale
    macro_image = np.uint8(macro_image * 255)
    _, binary_image = cv2.threshold(macro_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_macros = len(contours)

    # Create an image to draw the contours
    contour_image = np.zeros_like(binary_image)

    # Draw contours on a blank image
    cv2.drawContours(contour_image, contours, -1, (255), thickness=cv2.FILLED)

    # Compute the distance transform
    dist_transform = cv2.distanceTransform(contour_image, cv2.DIST_L2, 5)

    # Calculate connectivity by analyzing diagonal distances
    # Here we consider connectivity by looking for non-zero distances in the diagonal directions
    diagonal_kernel = np.array([[1, 0, 1],
                                [0, 0, 0],
                                [1, 0, 1]], dtype=np.uint8)

    diagonal_connectivity_map = cv2.filter2D((dist_transform > 0).astype(np.uint8), -1, diagonal_kernel)

    # Calculate diagonal connectivity as the number of adjacent pixels in diagonals
    diagonal_connectivity = np.sum(diagonal_connectivity_map > 0)

    # Convert the result to physical units
    feature_value_um = diagonal_connectivity * (tiles_size ** 2)

    return {"macro_diagonal_connectivity": feature_value_um}


def rudy_gradation_smoothness(images):
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

    # Compute gradients of the RUDY image
    rudy_dx = cv2.Sobel(rudy_image, cv2.CV_64F, 1, 0, ksize=3)
    rudy_dy = cv2.Sobel(rudy_image, cv2.CV_64F, 0, 1, ksize=3)

    # Compute the gradient magnitude
    rudy_magnitude = np.sqrt(rudy_dx**2 + rudy_dy**2)
    
    # Calculate the smoothness as the average of gradient magnitudes
    smoothness = np.mean(rudy_magnitude)
    
    return {"rudy_gradation_smoothness": smoothness}

def macro_edge_proximity_to_pins(images):
    tiles_size = 2.25
    macro_image = images[0]
    rudy_pin_image = images[2]
    
    # Convert macro image to [0-255]
    macro_image = np.uint8(macro_image * 255)
    
    # Binarize the macro image
    _, binary_image = cv2.threshold(macro_image, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours of macros
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Compute edge proximity
    edge_distances = []

    for contour in contours:
        for point in contour:
            # Get the coordinates of the edge point
            x, y = point[0]
            
            # Check the proximity to pin clusters
            if np.any(rudy_pin_image[max(0, y-1):min(y+2, rudy_pin_image.shape[0]),
                                     max(0, x-1):min(x+2, rudy_pin_image.shape[1])] > 0):
                edge_distances.append((x, y))
    
    # Calculate the average proximity
    average_proximity = len(edge_distances) * tiles_size
    
    return {"macro_edge_proximity_to_pins": average_proximity}


def macro_cluster_compactness(images):
    tiles_size = 2.25
    macro_image = images[0]
    
    image_height, image_width = macro_image.shape
    total_image_area = image_width * image_height * (tiles_size ** 2)  # Convert pixel area to um²
    
    # Convert macro image to 0-255 range
    macro_image = np.uint8(macro_image * 255)
    _, binary_image = cv2.threshold(macro_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calculate total macro area and perimeter
    total_macro_area = 0
    total_macro_perimeter = 0
    
    for contour in contours:
        area = cv2.contourArea(contour) * (tiles_size ** 2)  # Convert pixel area to um²
        perimeter = cv2.arcLength(contour, True) * tiles_size  # Convert pixel perimeter to um
        total_macro_area += area
        total_macro_perimeter += perimeter
    
    # Calculate compactness: (Perimeter^2 / (4 * π * Area))
    if total_macro_area > 0:
        compactness = (total_macro_perimeter ** 2) / (4 * np.pi * total_macro_area)
    else:
        compactness = 0  # Avoid division by zero
    
    feature_value = compactness
    return {"macro_cluster_compactness": feature_value}

def pin_density_variance(images):
    tiles_size = 2.25
    macro_image = images[0]
    rudy_image = images[1]
    rudy_pin_image = images[2]

    image_height, image_width = macro_image.shape
    total_image_area = image_width * image_height

    # Convert macro image for contour detection
    macro_image = np.uint8(macro_image * 255)
    _, binary_image = cv2.threshold(macro_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_macros = len(contours)

    # Calculate pin densities
    pin_densities = []

    for contour in contours:
        mask = np.zeros(rudy_pin_image.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, (1), thickness=cv2.FILLED)

        # Calculate pin density in this region
        pin_area = cv2.countNonZero(rudy_pin_image * mask)
        contour_area = cv2.contourArea(contour)
        pin_density = (pin_area / contour_area) if contour_area != 0 else 0
        pin_densities.append(pin_density)

    # Calculate variance of the pin densities
    pin_density_variance = np.var(pin_densities)

    return {"pin_density_variance": pin_density_variance}



def pin_neighborhood_uniformity(images):
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
    
    # Calculate pin density
    pin_density = np.sum(rudy_pin_image) / total_image_area
    
    # Calculate standard deviation of pin densities
    pin_density_map = rudy_pin_image * (255 / np.amax(rudy_pin_image))
    neighborhood_size = 5  # You can adjust this based on the desired neighborhood size
    pin_density_stddev = np.std([cv2.mean(pin_density_map[y:y+neighborhood_size, x:x+neighborhood_size])[0]
                                 for y in range(0, image_height, neighborhood_size)
                                 for x in range(0, image_width, neighborhood_size)])
    
    # Normalize the standard deviation by the maximum possible standard deviation
    max_possible_stddev = 255 / np.sqrt(total_image_area)
    uniformity = 1 - (pin_density_stddev / max_possible_stddev)
    
    return {"pin_neighborhood_uniformity": uniformity}


def rudy_consistency_index(images):
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

    # Calculate mean and standard deviation of RUDY values
    mean_rudy = np.mean(rudy_image)
    std_dev_rudy = np.std(rudy_image)

    # Calculate RUDY Consistency Index
    if mean_rudy != 0:
        consistency_index = std_dev_rudy / mean_rudy
    else:
        consistency_index = 0
    
    feature_value = consistency_index

    return {"rudy_consistency_index": feature_value}


def pin_to_macro_rudy_gradient_proximity(images):
    tiles_size = 2.25
    macro_image = images[0]
    rudy_image = images[1]
    rudy_pin_image = images[2]
    
    # Convert macro image to [0-255]
    macro_image = np.uint8(macro_image * 255)
    
    # Binarize macro image
    _, binary_image = cv2.threshold(macro_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Compute gradients of RUDY image
    grad_x = cv2.Sobel(rudy_image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(rudy_image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = cv2.magnitude(grad_x, grad_y)
    
    # Threshold gradients to find significant changes
    _, gradient_thresh = cv2.threshold(gradient_magnitude, 0.1, 1, cv2.THRESH_BINARY)
    
    # Overlay pin image
    combined_image = cv2.multiply(gradient_thresh, rudy_pin_image)
    
    # Calculate proximity by measuring overlap with macro regions
    proximity_score = 0
    for cnt in contours:
        mask = np.zeros_like(combined_image)
        cv2.drawContours(mask, [cnt], 0, 1, thickness=cv2.FILLED)
        overlap = cv2.multiply(combined_image, mask)
        proximity_score += np.sum(overlap)
    
    # Convert proximity score to real-world units
    proximity_score_um = proximity_score * (tiles_size ** 2)
    
    return {"pin_to_macro_rudy_gradient_proximity": proximity_score_um}

def sector_rudy_disparity(images):
    import numpy as np
    import cv2
    
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
    
    # Compute the RUDY disparity feature
    # Normalize RUDY image to [0, 1]
    rudy_normalized = np.array(rudy_image)
    
    # Divide the image into sectors (e.g., a grid of 4x4 for simplicity)
    sector_size = 64  # Assuming a 4x4 grid for the 256x256 image
    disparities = []
    
    for i in range(0, image_height, sector_size):
        for j in range(0, image_width, sector_size):
            # Extract the sector
            sector = rudy_normalized[i:i+sector_size, j:j+sector_size]
            
            # Calculate the average RUDY value for the sector
            sector_average = np.mean(sector)
            disparities.append(sector_average)
    
    # Calculate disparity as the variance or standard deviation of sector averages
    sector_rudy_disparity_value = np.std(disparities)
    
    # Convert the disparity value into correct units given:
    # Each pixel is 2.25 um x 2.25 um
    sector_rudy_disparity_um = sector_rudy_disparity_value * tiles_size
    
    return {"sector_rudy_disparity": sector_rudy_disparity_um}

def macro_corner_count(images):
    tiles_size = 2.25
    macro_image = images[0]
    rudy_image = images[1]
    rudy_pin_image = images[2]
    
    image_height, image_width = macro_image.shape
    total_image_area = image_width * image_height
    
    macro_image = np.uint8(macro_image * 255)
    _, binary_image = cv2.threshold(macro_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Convert contours to polygons and find corners
    num_corners = 0
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        num_corners += len(approx)
    
    feature_value = num_corners
    
    return {"macro_corner_count": feature_value}


def pin_to_macro_edge_proximity_std(images):
    tiles_size = 2.25
    macro_image = images[0]
    rudy_image = images[1]  # Not used in this function
    rudy_pin_image = images[2]
    
    image_height, image_width = macro_image.shape
    
    # Convert macro image to binary
    macro_image = np.uint8(macro_image * 255)
    _, binary_image = cv2.threshold(macro_image, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours of the macro regions
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a mask for the macro contours
    macro_mask = np.zeros_like(macro_image)
    cv2.drawContours(macro_mask, contours, -1, (255), thickness=cv2.FILLED)
    
    # Calculate distance from each point to the nearest macro edge
    dist_to_macro_edge = cv2.distanceTransform(255 - macro_mask, cv2.DIST_L2, 3)
    
    # Consider only pin regions from the RUDY pin image
    pin_indices = np.where(rudy_pin_image > 0.5)
    
    # Gather distances of pins to nearest macro edges
    pin_distances = dist_to_macro_edge[pin_indices]
    
    # Convert distances from pixels to um
    pin_distances_um = pin_distances * tiles_size
    
    # Calculate the standard deviation
    proximity_std = np.std(pin_distances_um)
    
    return {"pin_to_macro_edge_proximity_std": proximity_std}


def macro_linear_alignment(images):
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
    
    # Initialize alignment score
    alignment_score = 0
    
    for contour in contours:
        # Fit a bounding rectangle around each macro
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate aspect ratio of the rectangle
        aspect_ratio = w / float(h)
        
        # Check if the rectangle is linear (either vertical or horizontal)
        if aspect_ratio > 1.5 or aspect_ratio < 0.67:
            alignment_score += 1

    # Normalize the score by the number of macros
    if num_macros > 0:
        alignment_score /= num_macros
    
    # Calculate the feature value in micrometers
    feature_value = alignment_score * tiles_size
    
    return {"macro_linear_alignment": feature_value}


def rudy_peak_clustering(images):
    tiles_size = 2.25
    macro_image = images[0]
    rudy_image = images[1]
    rudy_pin_image = images[2]
    
    image_height, image_width = macro_image.shape
    total_image_area = image_width * image_height
    
    # Convert macro image for contour detection
    macro_image = np.uint8(macro_image * 255)
    _, binary_image = cv2.threshold(macro_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_macros = len(contours)

    # Identify RUDY peaks
    rudy_threshold_value = 0.8  # Example value, tune as necessary
    _, rudy_peaks = cv2.threshold(rudy_image, rudy_threshold_value, 1, cv2.THRESH_BINARY)

    # Detect connected components (clusters) in RUDY peaks
    num_labels, labels_im = cv2.connectedComponents(np.uint8(rudy_peaks))

    # Calculate clustering metric
    peak_clusters = [np.sum(labels_im == i) for i in range(1, num_labels)]
    clustering_score = np.sum(np.array(peak_clusters) ** 2) / total_image_area

    return {"rudy_peak_clustering": clustering_score * (tiles_size ** 2)}


def macro_pin_alignment_score(images):
    tiles_size = 2.25
    macro_image = images[0]
    rudy_image = images[1]
    rudy_pin_image = images[2]
    
    image_height, image_width = macro_image.shape
    total_image_area = image_width * image_height
    
    # Convert macro image to 8-bit
    macro_image = np.uint8(macro_image * 255)
    _, binary_image = cv2.threshold(macro_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_macros = len(contours)
    
    # Calculate macro boundaries
    macro_boundaries = [cv2.boundingRect(cnt) for cnt in contours]
    
    # Calculate pin clusters
    _, rudy_pin_thresh = cv2.threshold(rudy_pin_image, 0.5, 1, cv2.THRESH_BINARY)
    rudy_pin_thresh = np.uint8(rudy_pin_thresh * 255)
    pin_contours, _ = cv2.findContours(rudy_pin_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    pin_boundaries = [cv2.boundingRect(cnt) for cnt in pin_contours]
    
    # Calculate alignment score
    alignment_score = 0
    for macro in macro_boundaries:
        macro_x, macro_y, macro_w, macro_h = macro
        for pin in pin_boundaries:
            pin_x, pin_y, pin_w, pin_h = pin
            
            # Check if the pin cluster is aligned with the macro boundary
            if (macro_x <= pin_x <= macro_x + macro_w) and (macro_y <= pin_y <= macro_y + macro_h):
                pin_area = pin_w * pin_h
                alignment_score += pin_area

    # Convert score to um^2
    alignment_score *= (tiles_size ** 2)

    return {"macro_pin_alignment_score": alignment_score}


def pin_density_gradient(images):
    tiles_size = 2.25
    macro_image = images[0]
    rudy_image = images[1]
    rudy_pin_image = images[2]
    
    # Convert macro image to [0-255]
    macro_image = np.uint8(macro_image * 255)
    
    # Threshold to create a binary image
    _, binary_image = cv2.threshold(macro_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_macros = len(contours)
    
    # Calculate the gradient of the RUDY pin image
    sobelx = cv2.Sobel(rudy_pin_image, cv2.CV_64F, 1, 0, ksize=5)  # Gradient in x-direction
    sobely = cv2.Sobel(rudy_pin_image, cv2.CV_64F, 0, 1, ksize=5)  # Gradient in y-direction
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # Normalize the gradient magnitude to represent density change
    max_gradient = np.max(gradient_magnitude)
    if max_gradient > 0:
        normalized_gradient = gradient_magnitude / max_gradient
    else:
        normalized_gradient = gradient_magnitude
    
    # Calculate the pin density gradient
    pin_density_gradient_value = np.sum(normalized_gradient) * (tiles_size**2)
    
    return {"pin_density_gradient": pin_density_gradient_value}

def macro_to_pin_cluster_proximity(images):
    tiles_size = 2.25  # size of each pixel in micrometers
    macro_image = images[0]
    rudy_image = images[1]
    rudy_pin_image = images[2]
    
    image_height, image_width = macro_image.shape
    total_image_area = image_width * image_height
    
    # Convert macro image to [0, 255]
    macro_image = np.uint8(macro_image * 255)
    _, binary_image = cv2.threshold(macro_image, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours in the macro image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_macros = len(contours)
    
    # Find pin clusters (bright areas) in the RUDY pin image
    _, pin_clusters = cv2.threshold(np.uint8(rudy_pin_image * 255), 127, 255, cv2.THRESH_BINARY)
    pin_centroids = cv2.connectedComponentsWithStats(pin_clusters, connectivity=8)[3][1:]  # skip the background
    
    # Calculate average proximity distance from each macro to pin clusters
    total_distance = 0
    num_distances = 0
    
    for macro_contour in contours:
        macro_centroid = np.mean(macro_contour, axis=0)[0]  # Get the centroid of the macro
        for pin_centroid in pin_centroids:
            distance_pixels = np.linalg.norm(macro_centroid - pin_centroid)
            distance_um = distance_pixels * tiles_size
            total_distance += distance_um
            num_distances += 1
    
    # Avoid division by zero
    if num_distances > 0:
        average_proximity_distance = total_distance / num_distances
    else:
        average_proximity_distance = 0
    
    return {"macro_to_pin_cluster_proximity": average_proximity_distance}


feat_func_list = [macro_spacing_std,
 macro_boundary_distance_var,
 pin_clustering_factor,
 macro_diagonal_connectivity,
 rudy_gradation_smoothness,
 macro_edge_proximity_to_pins,
 macro_cluster_compactness,
 pin_density_variance,
 pin_neighborhood_uniformity,
 rudy_consistency_index,
 pin_to_macro_rudy_gradient_proximity,
 sector_rudy_disparity,
 macro_corner_count,
 pin_to_macro_edge_proximity_std,
 macro_linear_alignment,
 rudy_peak_clustering,
 macro_pin_alignment_score,
 pin_density_gradient,
 macro_to_pin_cluster_proximity]