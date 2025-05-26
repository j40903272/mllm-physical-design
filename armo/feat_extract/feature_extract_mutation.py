import numpy as np
import cv2

feat_pool = {'average_macro_angular_centrality': 'the average angular alignment of macros relative to the layout center, indicating placement symmetry and routing balance',
 'high_density_rudy_pin_ratio': 'the ratio of hotspots area in the rudy pin map',
 'mean_macro_neighborhood_density': 'the average density of macros within defined neighborhood regions, indicating local macro clustering',
 'macro_congestion_potential_index': 'a measure of potential congestion around macros, calculated by evaluating the intersection density of rudy map hotspots with macro areas, indicating areas where macros might contribute to routing congestion due to overlapping high-demand regions',
 'mean_rudy_gradient_magnitude': 'the average magnitude of the gradient in the RUDY map, indicating the overall intensity of routing demand shifts across the layout',
 'max_macro_contact_density': 'the maximum density of direct contact between macro edges, indicating areas with high macro interaction and potential congestion',
 'mean_rudy_pin_centrality': 'the average centrality of RUDY pins within clusters, indicating the likelihood of routing density peaking at cluster centers',
 'macro_gradient_variation_index': 'a measure of the variation in density gradients between macro regions, indicating uneven congestion distribution',
 'mean_rudy_pin': 'the average of the rudy pin map',
 'mean_macro_central_congestion_intensity': 'the average intensity of congestion within the central region of macros, calculated by evaluating the overlap of high-demand areas in the RUDY map with central macro zones, indicating potential routing pressure in core macro areas',
 'mean_macro_curvature_variance': 'the variance in curvature of macro boundaries, indicating irregularities in macro shapes affecting routing',
 'high_density_rudy_ratio': 'the ratio of hotspots area in the rudy map',
 'mean_macro_radial_convergence': "the average radial alignment of macros towards central regions, indicating potential congestion in the layout's core",
 'PAR_rudy_pin': 'the Peak-to-Average Ratio of the rudy pin map',
 'macro_neighbor_proximity_variance': 'the variance in distances between neighboring macros, indicating irregular macro placement patterns',
 'mean_macro_core_to_rudy_edge_distance': 'the average distance from the core region of macros to the nearest high-demand edge in the RUDY map, indicating potential routing bottlenecks near macro cores',
 'std_rudy_pin': 'the standard deviation of the rudy pin map',
 'mean_macro_intersection_intensity': 'the average overlap intensity of macros, indicating the density of intersections between macro boundaries',
 'mean_macro_cluster_compactness': 'the average measure of how tightly grouped macro clusters are in the layout, potentially impacting congestion within clusters',
 'mean_macro_rectilinearity': 'the average rectilinear shape factor of macros, calculated as the ratio of macro perimeter to its bounding box perimeter, indicating macro alignment to grid structures'}

def average_macro_angular_centrality(images):
    tiles_size = 2.25
    macro_image = images[0]
    rudy_image = images[1]
    rudy_pin_image = images[2]
    
    image_height, image_width = macro_image.shape
    total_image_area = image_width * image_height
    
    # Convert macro image to [0, 255] range
    macro_image = np.uint8(macro_image * 255)
    
    # Threshold and find contours
    _, binary_image = cv2.threshold(macro_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calculate the layout centroid
    layout_centroid = (image_width / 2, image_height / 2)
    
    # Calculate the centroids of the macros
    macro_centroids = [cv2.moments(contour) for contour in contours]
    macro_centroids = [
        (int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])) for m in macro_centroids if m["m00"] != 0
    ]
    
    # Calculate angular displacement
    angles = []
    for centroid in macro_centroids:
        dx = centroid[0] - layout_centroid[0]
        dy = centroid[1] - layout_centroid[1]
        angle = np.arctan2(dy, dx)
        angles.append(angle)

    # Calculate the average angular centrality
    average_angular_centrality = np.mean(angles)
    
    return {"average_macro_angular_centrality": np.degrees(average_angular_centrality)}

def high_density_rudy_pin_ratio(images):
    image = images[2]
    total_area = image.shape[0] * image.shape[1]
    mean_rudy_pin = np.mean(image)
    high_density_rudy_pin_ratio = (image > mean_rudy_pin).sum() /  total_area
    
    return {
        "high_density_rudy_pin_ratio": high_density_rudy_pin_ratio,
    }
    

def mean_macro_neighborhood_density(images):
    tiles_size = 2.25
    macro_image = images[0]
    
    # Convert macro image to [0, 255]
    macro_image = np.uint8(macro_image * 255)

    # Convert macro image to binary
    _, binary_image = cv2.threshold(macro_image, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours which represent macros
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Set the neighborhood radius in pixels
    neighborhood_radius = 10  # Example size; adjust as needed
    
    # Create a mask for calculating macro density
    macro_mask = np.zeros_like(macro_image)
    
    # Draw the contours onto the mask
    cv2.drawContours(macro_mask, contours, -1, (255), thickness=cv2.FILLED)
    
    # Calculate macro density
    kernel_size = 2 * neighborhood_radius + 1
    macro_area = np.sum(macro_mask > 0)
    total_area = macro_mask.shape[0] * macro_mask.shape[1]
    
    # Calculate neighborhood density using a box filter
    neighborhood_area = cv2.boxFilter(macro_mask, ddepth=cv2.CV_64F, ksize=(kernel_size, kernel_size), normalize=False)
    
    # Convert pixel area to micrometer square
    pixel_area_um2 = tiles_size * tiles_size
    macro_area_um2 = macro_area * pixel_area_um2
    neighborhood_area_um2 = neighborhood_area * pixel_area_um2
    
    # Calculate average macro density within the neighborhood
    mean_density = np.mean(neighborhood_area_um2[macro_mask > 0]) / (kernel_size * kernel_size * pixel_area_um2)
    
    return {"mean_macro_neighborhood_density": mean_density}

def macro_congestion_potential_index(images):
    tiles_size = 2.25
    macro_image = images[0]
    rudy_image = images[1]
    rudy_pin_image = images[2]
    
    image_height, image_width = macro_image.shape
    total_image_area = image_width * image_height
    
    # Convert the macro image to [0, 255] scale
    macro_image = np.uint8(macro_image * 255)
    _, binary_image = cv2.threshold(macro_image, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours in the macro image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_macros = len(contours)
    
    # Calculate overlap area
    overlap_area = 0.0
    for contour in contours:
        # Create a mask for the current macro
        macro_mask = np.zeros_like(rudy_image)
        cv2.drawContours(macro_mask, [contour], -1, 1, thickness=cv2.FILLED)
        
        # Calculate the intersection density
        overlap = np.sum(rudy_image * macro_mask)
        overlap_area += overlap
    
    # Convert overlap area from pixels to square micrometers
    overlap_area_um2 = overlap_area * (tiles_size ** 2)

    # Normalize by the number of macros and image area
    macro_congestion_potential = overlap_area_um2 / (num_macros * total_image_area * (tiles_size ** 2))
    
    return {"macro_congestion_potential_index": macro_congestion_potential}


def mean_rudy_gradient_magnitude(images):
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
    
    # Calculate gradient magnitude of RUDY map
    grad_x = cv2.Sobel(rudy_image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(rudy_image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Calculate the mean
    mean_gradient_magnitude = np.mean(gradient_magnitude)
    
    # Convert mean gradient magnitude to the appropriate units (um)
    mean_gradient_magnitude_um = mean_gradient_magnitude * tiles_size
    
    return {"mean_rudy_gradient_magnitude": mean_gradient_magnitude_um}


def max_macro_contact_density(images):
    tiles_size = 2.25
    macro_image = images[0]
    rudy_image = images[1]
    rudy_pin_image = images[2]
    
    image_height, image_width = macro_image.shape
    total_image_area = image_width * image_height
    
    # Convert macro image to [0, 255]
    macro_image = np.uint8(macro_image * 255)
    _, binary_image = cv2.threshold(macro_image, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours in the binary macro image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calculate area occupied by each macro
    macro_areas = [cv2.contourArea(cnt) for cnt in contours]
    
    # Calculate contact density
    max_density = 0
    for area in macro_areas:
        # Convert pixel area to square micrometers
        area_um2 = area * (tiles_size**2)
        # Compute density as the area occupied by macros in squared micrometers
        density = area_um2 / total_image_area
        # Update maximum density if current density is higher
        if density > max_density:
            max_density = density
    
    # Create the feature dictionary
    feature_name = 'max_macro_contact_density'
    feature_value = max_density
    
    return {feature_name: feature_value}


def mean_rudy_pin_centrality(images):
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
    
    # Calculate the centroid of the layout
    centroid_x = image_width / 2
    centroid_y = image_height / 2

    # Find areas of high-density Rudy pin values
    # Assuming high-density is above a certain threshold, e.g., 0.5
    high_density_threshold = 0.5
    high_density_positions = np.argwhere(rudy_pin_image > high_density_threshold)

    # Calculate distances from high-density areas to the centroid
    distances = []
    for pos in high_density_positions:
        y, x = pos
        distance = np.sqrt((x - centroid_x) ** 2 + (y - centroid_y) ** 2)
        distances.append(distance * tiles_size)  # Convert to micrometers

    # Calculate mean centrality
    if len(distances) > 0:
        mean_centrality = np.mean(distances)
    else:
        mean_centrality = 0

    return {"mean_rudy_pin_centrality": mean_centrality}

def macro_gradient_variation_index(images):
    tiles_size = 2.25
    macro_image = images[0]
    rudy_image = images[1]
    rudy_pin_image = images[2]
    
    image_height, image_width = macro_image.shape
    total_image_area = image_width * image_height
    
    # Scale the macro image to [0, 255]
    macro_image = np.uint8(macro_image * 255)
    _, binary_image = cv2.threshold(macro_image, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours to determine the number of macros
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_macros = len(contours)
    
    # Calculate gradients
    gradient_x = cv2.Sobel(macro_image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(macro_image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = cv2.magnitude(gradient_x, gradient_y)
    
    # Calculate standard deviation of gradient magnitudes
    macro_gradient_variation_index_value = np.std(gradient_magnitude)
    
    # Convert to appropriate area unit (square micrometers)
    macro_gradient_variation_index_um = macro_gradient_variation_index_value * tiles_size
    
    return {"macro_gradient_variation_index": macro_gradient_variation_index_um}


def mean_rudy_pin(images):
    image = images[2]
    max_rudy = np.max(image)
    min_rudy = np.min(image)
    mean_rudy = np.mean(image)
    
    return {
        "mean_rudy_pin": mean_rudy,
    }
    
    
def mean_macro_central_congestion_intensity(images):
    tiles_size = 2.25
    macro_image = images[0]
    rudy_image = images[1]
    rudy_pin_image = images[2]
    
    # Convert the macro image to 0-255
    macro_image = np.uint8(macro_image * 255)
    
    # Threshold macro image
    _, binary_image = cv2.threshold(macro_image, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours of the macro blocks
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    central_congestion_intensity = []

    for contour in contours:
        # Calculate bounding box of the macro
        x, y, w, h = cv2.boundingRect(contour)

        # Identify central region (e.g., you may define it as the center 50% of each dimension)
        central_x_start = int(x + 0.25 * w)
        central_y_start = int(y + 0.25 * h)
        central_x_end = int(x + 0.75 * w)
        central_y_end = int(y + 0.75 * h)

        # Crop the RUDY map to the central region of the macro
        central_region_rudy = rudy_image[central_y_start:central_y_end, central_x_start:central_x_end]

        # Calculate mean intensity of the central region
        mean_intensity = np.mean(central_region_rudy)
        central_congestion_intensity.append(mean_intensity)

    # Calculate the mean congestion intensity for all central macro regions
    feature_value = np.mean(central_congestion_intensity)

    # Note: Intensity is already normalized [0-1] so no need to multiply with area units
    return {"mean_macro_central_congestion_intensity": feature_value}


def mean_macro_curvature_variance(images):
    tiles_size = 2.25
    macro_image = images[0]
    rudy_image = images[1]
    rudy_pin_image = images[2]
    
    image_height, image_width = macro_image.shape
    total_image_area = image_width * image_height
    
    # Convert macro image to uint8 and apply binary threshold
    macro_image = np.uint8(macro_image * 255)
    _, binary_image = cv2.threshold(macro_image, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_macros = len(contours)
    
    # Calculate curvature variance for each macro
    def find_curvature(contour):
        contour = contour.squeeze()
        dx = np.gradient(contour[:, 0])
        dy = np.gradient(contour[:, 1])
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        curvature = np.abs(ddx * dy - dx * ddy) / np.power(dx * dx + dy * dy, 3/2 + np.finfo(float).eps)
        return np.var(curvature)
    
    curvature_variances = [find_curvature(contour) for contour in contours]
    
    # Calculate the mean curvature variance
    if num_macros > 0:
        mean_curvature_variance = np.mean(curvature_variances)
    else:
        mean_curvature_variance = 0
    
    feature_value = mean_curvature_variance
    
    return {"mean_macro_curvature_variance": feature_value}


def high_density_rudy_ratio(images):
    image = images[1]
    total_area = image.shape[0] * image.shape[1]
    mean_rudy = np.mean(image)
    high_density_rudy_ratio = (image > mean_rudy).sum() /  total_area
    
    return {
        "high_density_rudy_ratio": high_density_rudy_ratio,
    }

def mean_macro_radial_convergence(images):
    tiles_size = 2.25
    macro_image = images[0]
    
    # Convert macro image to [0, 255]
    macro_image = np.uint8(macro_image * 255)
    
    # Convert to binary image
    _, binary_image = cv2.threshold(macro_image, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours of the macros
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_macros = len(contours)

    # Compute centroids of macros
    centroids = []
    for contour in contours:
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
            centroids.append((cX, cY))
    
    # Calculate central point, assuming it is the center of the image
    image_height, image_width = macro_image.shape
    central_point = (image_width // 2, image_height // 2)
    
    # Calculate mean angular difference from the central radial axis (horizontal axis)
    angular_differences = []
    for (cX, cY) in centroids:
        vector = (cX - central_point[0], cY - central_point[1])
        angle = np.arctan2(vector[1], vector[0])  # Angle from horizontal axis
        
        # Calculate angular difference from radial axis (0 or a chosen axis if defined)
        angular_difference = abs(angle)  # Absolute value of angle difference
        angular_differences.append(angular_difference)
    
    mean_angular_difference = np.mean(angular_differences)
    
    # Convert to micrometers
    mean_angular_difference_um = mean_angular_difference * tiles_size
    
    return {"mean_macro_radial_convergence": mean_angular_difference_um}


def PAR_rudy_pin(images):
    image = images[2]
    total_area = image.shape[0] * image.shape[1]
    max_rudy = np.max(image)
    min_rudy = np.min(image)
    mean_rudy = np.mean(image)
    std_rudy = np.std(image)
    par_rudy = max_rudy / mean_rudy
    high_density_rudy_ratio = (image > mean_rudy).sum() /  total_area
    
    return {
        "PAR_rudy_pin": par_rudy,
    }
    
    
def std_rudy_pin(images):
    image = images[2]
    total_area = image.shape[0] * image.shape[1]
    max_rudy = np.max(image)
    min_rudy = np.min(image)
    mean_rudy = np.mean(image)
    std_rudy = np.std(image)
    par_rudy = max_rudy / mean_rudy
    high_density_rudy_ratio = (image > mean_rudy).sum() /  total_area
    
    return {
        "std_rudy_pin": std_rudy,
    }
    
    
def macro_neighbor_proximity_variance(images):
    tiles_size = 2.25
    macro_image = images[0]
    rudy_image = images[1]
    rudy_pin_image = images[2]
    
    image_height, image_width = macro_image.shape
    total_image_area = image_width * image_height
    
    # Convert the macro image to uint8 and apply binary thresholding
    macro_image = np.uint8(macro_image * 255)
    _, binary_image = cv2.threshold(macro_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_macros = len(contours)
    
    # Calculate centroids of each macro
    centroids = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            c_x = int(M["m10"] / M["m00"])
            c_y = int(M["m01"] / M["m00"])
            centroids.append((c_x, c_y))
    
    # Calculate pairwise distances between centroids
    distances = []
    for i in range(len(centroids)):
        for j in range(i+1, len(centroids)):
            dist = np.sqrt((centroids[i][0] - centroids[j][0])**2 + (centroids[i][1] - centroids[j][1])**2)
            # Convert distance to micrometers
            dist_um = dist * tiles_size
            distances.append(dist_um)
    
    # Calculate variance of the distances
    if len(distances) > 0:
        variance = np.var(distances)
    else:
        variance = 0.0

    return {"macro_neighbor_proximity_variance": variance}


def mean_macro_core_to_rudy_edge_distance(images):
    tiles_size = 2.25
    macro_image = images[0]
    rudy_image = images[1]
    rudy_pin_image = images[2]
    
    # Convert macro image to 0-255 scale
    macro_image = np.uint8(macro_image * 255)
    _, binary_image = cv2.threshold(macro_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Threshold RUDY image to identify high-demand regions
    rudy_threshold = 0.5
    _, rudy_edges = cv2.threshold(np.float32(rudy_image), rudy_threshold, 1, cv2.THRESH_BINARY)
    rudy_edges = np.uint8(rudy_edges * 255)  # Convert for distance transform
    dist_transform = cv2.distanceTransform(cv2.bitwise_not(rudy_edges), cv2.DIST_L2, 3)

    # Calculate distances from macro cores to nearest high-demand edge
    distances = []
    for contour in contours:
        if cv2.contourArea(contour) > 0:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                distances.append(dist_transform[cy, cx])

    # Convert pixel distance to micrometers
    distances_um = [d * tiles_size for d in distances]

    # Calculate mean distance
    mean_distance = np.mean(distances_um) if distances_um else 0

    return {"mean_macro_core_to_rudy_edge_distance": mean_distance}

def mean_macro_intersection_intensity(images):
    tiles_size = 2.25
    macro_image = images[0]
    rudy_image = images[1]
    rudy_pin_image = images[2]
    
    # Convert macro image to 0-255 scale
    macro_image = np.uint8(macro_image * 255)
    
    # Threshold to obtain binary image of macros
    _, binary_image = cv2.threshold(macro_image, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours of the macros
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_macros = len(contours)

    # Create a mask from the contours
    macro_mask = np.zeros_like(macro_image)
    cv2.drawContours(macro_mask, contours, -1, 255, thickness=cv2.FILLED)
    
    # Calculate intersections with RUDY and RUDY pin
    intersection_rudy = cv2.bitwise_and(macro_mask, np.uint8(rudy_image * 255))
    intersection_rudy_pin = cv2.bitwise_and(macro_mask, np.uint8(rudy_pin_image * 255))
    
    # Count non-zero pixels for each intersection
    count_intersection_rudy = cv2.countNonZero(intersection_rudy)
    count_intersection_rudy_pin = cv2.countNonZero(intersection_rudy_pin)
    
    # Calculate area in um^2
    area_per_pixel = tiles_size * tiles_size
    image_height, image_width = macro_image.shape
    total_image_area = image_width * image_height
    total_area_um2 = total_image_area * area_per_pixel
    
    # Calculate mean intersection intensity per um^2
    mean_intensity = (count_intersection_rudy + count_intersection_rudy_pin) / total_area_um2
    
    return {"mean_macro_intersection_intensity": mean_intensity}

def mean_macro_cluster_compactness(images):
    tiles_size = 2.25
    macro_image = images[0]
    rudy_image = images[1]
    rudy_pin_image = images[2]
    
    # Convert macro image to [0-255]
    macro_image = np.uint8(macro_image * 255)
    
    # Find contours of the macros in the binary image
    _, binary_image = cv2.threshold(macro_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    compactness_list = []

    for contour in contours:
        # Calculate the area of the macro
        area = cv2.contourArea(contour)
        if area == 0:
            continue
        
        # Calculate the minimum enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(contour)
        circle_area = np.pi * (radius ** 2)
        
        # Calculate compactness
        compactness = area / circle_area
        compactness_list.append(compactness)
    
    # Calculate the mean compactness
    if compactness_list:
        mean_compactness = np.mean(compactness_list)
    else:
        mean_compactness = 0
    
    # Convert from pixel units to micrometer units
    feature_value = mean_compactness

    return {"mean_macro_cluster_compactness": feature_value}


def mean_macro_rectilinearity(images):
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

    total_rectilinearity = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if area > 0:
            rectilinearity = perimeter / area
            total_rectilinearity += rectilinearity
    
    if num_macros > 0:
        mean_rectilinearity = total_rectilinearity / num_macros
    else:
        mean_rectilinearity = 0

    # Convert from pixel units to micrometers
    mean_rectilinearity = mean_rectilinearity / tiles_size

    return {"mean_macro_rectilinearity": mean_rectilinearity}

    
feat_func_list = [average_macro_angular_centrality,
 high_density_rudy_pin_ratio,
 mean_macro_neighborhood_density,
 macro_congestion_potential_index,
 mean_rudy_gradient_magnitude,
 max_macro_contact_density,
 mean_rudy_pin_centrality,
 macro_gradient_variation_index,
 mean_rudy_pin,
 mean_macro_central_congestion_intensity,
 mean_macro_curvature_variance,
 high_density_rudy_ratio,
 mean_macro_radial_convergence,
 PAR_rudy_pin,
 macro_neighbor_proximity_variance,
 mean_macro_core_to_rudy_edge_distance,
 std_rudy_pin,
 mean_macro_intersection_intensity,
 mean_macro_cluster_compactness,
 mean_macro_rectilinearity]