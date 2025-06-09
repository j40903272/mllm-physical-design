import numpy as np
import cv2
from scipy.stats import kurtosis
from scipy.stats import skew

feat_pool = {'max_congestion_ripple': 'the maximum observed propagation of congestion from a source point',
 'congestion_gradient': 'the rate of change in congestion levels across the layout, highlighting potential bottleneck areas',
 'mean_macro_proximity': 'the average distance between macros and nearby elements, indicating potential congestion zones',
 'diagonal_cell_density_gradient': 'the rate of change in cell density along the diagonals of the layout, indicating unique transition areas that may impact diagonal routing paths',
 'mean_cell_density_fluctuation': 'the average fluctuation of cell density in specific regions of the layout, highlighting potential areas of instability impacting routing paths and DRC violations',
 'congestion_transition_amplitude': 'the magnitude of change in congestion levels between adjacent regions, highlighting areas with significant transitions that could lead to potential routing bottlenecks or DRC violations.',
 'cell_density_variance_gradient': 'the variation in the standard deviation of cell density across different regions of the layout, indicating potential congestion transition zones',
 'congestion_variability_throughout_hierarchy': 'the variability of congestion levels across different hierarchical layers of the layout, identifying areas where hierarchical design decisions impact routing efficiency and DRC violation risk',
 'cell_density_skewness': 'the skewness of cell density distribution across the layout, indicating asymmetry and potential areas of congestion buildup',
 'macro_transition_band': 'the area surrounding macros where there is a rapid change in features like cell density or RUDY values, highlighting potential zones of congestion transition.',
 'cell_density_skewness_gradient': 'the rate of change in skewness of cell density across the layout, highlighting regions with asymmetric density distributions that could lead to routing challenges and affect DRC violations',
 'macro_interaction_perimeter': 'the total perimeter length of interaction zones between macros, indicating potential congestion and routing complexity at macro boundaries',
 'macro_interference_zone': 'the area around a macro where it likely interferes with cell placement and signal routing, increasing congestion risk',
 'cell_density_dipole': 'the presence of paired regions with significantly different cell densities, creating a dipolar distribution and potentially influencing routing paths and congestion behavior',
 'macro_compactness_index': 'the ratio of the perimeter to the area of macros in the layout, indicating the compactness and its potential impact on routing congestion',
 'cell_density_anisotropy': 'the directional variation in cell density across the layout, highlighting specific pathways with uneven distributions and potential routing constraints',
 'congestion_pressure_fluctuation': 'the variability in congestion pressure zones over time, highlighting areas with dynamic congestion behavior that may require adaptive routing strategies',
 'mean_eGR_local_adjacent_cohesion': 'the measure of cohesion in congestion overflow in early global routing between adjacent local regions, indicating areas where routing demand transitions smoothly or abruptly',
 'mean_eGR_local_variability': 'the variance of the mean congestion overflow in early global routing across local regions, identifying areas with inconsistent routing demands',
 'cell_density_fluctuation_balance': 'the balance or ratio of regions with increasing versus decreasing cell density fluctuations, indicating stability and potential impact areas on routing paths and DRC violations'}


def max_congestion_ripple(images):
    tiles_size = 2.25
    macro_image = images[0]
    cell_density_image = images[1]
    rudy_long_image = images[2]
    rudy_short_image = images[3]
    rudy_pin_long_image = images[4]
    congestion_eGR_horizontal_overflow_image = images[5]
    congestion_eGR_vertical_overflow_image = images[6]
    congestion_GR_horizontal_overflow_image = images[7]
    congestion_GR_vertical_overflow_image = images[8]
    
    image_height, image_width = macro_image.shape
    total_image_area = image_width * image_height
    
    macro_image = np.uint8(macro_image * 255)
    _, binary_image = cv2.threshold(macro_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_macros = len(contours)
    
    # Combine congestion overflow images
    combined_congestion_image = np.maximum.reduce([
        congestion_eGR_horizontal_overflow_image, 
        congestion_eGR_vertical_overflow_image,
        congestion_GR_horizontal_overflow_image,
        congestion_GR_vertical_overflow_image
    ])
    
    # Threshold to identify high congestion areas
    _, congestion_binary = cv2.threshold(combined_congestion_image, 0.5, 1.0, cv2.THRESH_BINARY)
    
    # Find contours in the congestion map
    congestion_binary_uint8 = np.uint8(congestion_binary * 255)
    contours, _ = cv2.findContours(congestion_binary_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    max_ripple_distance = 0
    
    for contour in contours:
        # Calculate the bounding rectangle for each congested area
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate the longest distance from the source in micrometers
        ripple_distance = max(w, h) * tiles_size
        max_ripple_distance = max(max_ripple_distance, ripple_distance)
    
    return {"max_congestion_ripple": max_ripple_distance}

def macro_interference_zone(images):
    tiles_size = 2.25
    macro_image = images[0]
    cell_density_image = images[1]
    rudy_long_image = images[2]
    rudy_short_image = images[3]
    rudy_pin_long_image = images[4]
    congestion_eGR_horizontal_overflow_image = images[5]
    congestion_eGR_vertical_overflow_image = images[6]
    congestion_GR_horizontal_overflow_image = images[7]
    congestion_GR_vertical_overflow_image = images[8]

    image_height, image_width = macro_image.shape
    total_image_area = image_width * image_height

    macro_image = np.uint8(macro_image * 255)
    _, binary_image = cv2.threshold(macro_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_macros = len(contours)

    # Calculating the macro interference zone.
    macro_interference_area_pixels = 0

    # Define the distance around the macros to consider as interference zone (example: 10 pixels)
    interference_zone_distance = 10

    for cnt in contours:
        # Get the bounding box of the current macro
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Expand the bounding box by the interference_zone_distance
        x_start = max(x - interference_zone_distance, 0)
        y_start = max(y - interference_zone_distance, 0)
        x_end = min(x + w + interference_zone_distance, image_width)
        y_end = min(y + h + interference_zone_distance, image_height)
        
        # Calculate the area over which this interference zone extends
        interference_zone_area = (x_end - x_start) * (y_end - y_start)
        macro_interference_area_pixels += interference_zone_area

    # Convert pixel area to physical area in um^2
    macro_interference_area_um2 = macro_interference_area_pixels * (tiles_size ** 2)

    return {"macro_interference_zone": macro_interference_area_um2}


def macro_compactness_index(images):
    tiles_size = 2.25
    macro_image = images[0]
    
    # Convert macro image to [0-255]
    macro_image = np.uint8(macro_image * 255)
    
    # Threshold to create a binary image
    _, binary_image = cv2.threshold(macro_image, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours of the macros
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize perimeter and area
    total_perimeter = 0.0
    total_area = 0.0

    # Calculate total perimeter and area of all macros
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        total_perimeter += perimeter
        total_area += area

    # Convert area from pixels to um^2 and perimeter to um
    total_area_um2 = total_area * (tiles_size ** 2)
    total_perimeter_um = total_perimeter * tiles_size

    # Calculate compactness index
    if total_area_um2 > 0:
        compactness_index = total_perimeter_um / total_area_um2
    else:
        compactness_index = 0

    return {"macro_compactness_index": compactness_index}

def cell_density_variance_gradient(images):
    tiles_size = 2.25
    macro_image = images[0]
    cell_density_image = images[1]
    rudy_long_image = images[2]
    rudy_short_image = images[3]
    rudy_pin_long_image = images[4]
    congestion_eGR_horizontal_overflow_image = images[5]
    congestion_eGR_vertical_overflow_image = images[6]
    congestion_GR_horizontal_overflow_image = images[7]
    congestion_GR_vertical_overflow_image = images[8]
    
    image_height, image_width = macro_image.shape
    total_image_area = image_width * image_height
    
    macro_image = np.uint8(macro_image * 255)
    _, binary_image = cv2.threshold(macro_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_macros = len(contours)
    
    # Calculate variance in standard deviation of cell density in different regions
    num_regions = 4  # Define how many regions you want, 4x4 grid in this case
    region_height = image_height // num_regions
    region_width = image_width // num_regions
    
    std_devs = []
    
    for i in range(num_regions):
        for j in range(num_regions):
            # Define the region of interest
            roi = cell_density_image[i * region_height: (i + 1) * region_height,
                                     j * region_width: (j + 1) * region_width]
            
            # Calculate standard deviation for the region
            std_dev = np.std(roi)
            std_devs.append(std_dev)
    
    # Calculate the variance of these standard deviations
    variance_of_std_dev = np.var(std_devs)
    
    return {"cell_density_variance_gradient": variance_of_std_dev}

def mean_macro_proximity(images):
    tiles_size = 2.25
    macro_image = images[0]
    cell_density_image = images[1]
    
    # Initialize macro image
    macro_image = np.uint8(macro_image * 255)
    _, binary_image = cv2.threshold(macro_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    total_distance = 0
    num_elements = 0
    
    # Iterate through each macro contour
    for cnt in contours:
        # Compute moments to find a center of macro
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        # Calculate proximity to nearby elements in cell density image
        nearby_pixels = cell_density_image[max(cY - 1, 0):min(cY + 2, cell_density_image.shape[0]),
                                           max(cX - 1, 0):min(cX + 2, cell_density_image.shape[1])]
        
        total_distance += np.sum(nearby_pixels) * tiles_size  # Accumulate distances considering pixel scale
        num_elements += np.count_nonzero(nearby_pixels)

    # Calculate mean proximity
    mean_proximity = total_distance / num_elements if num_elements else 0

    return {"mean_macro_proximity": mean_proximity}

def congestion_gradient(images):
    tiles_size = 2.25
    macro_image = images[0]
    cell_density_image = images[1]
    rudy_long_image = images[2]
    rudy_short_image = images[3]
    rudy_pin_long_image = images[4]
    congestion_eGR_horizontal_overflow_image = images[5]
    congestion_eGR_vertical_overflow_image = images[6]
    congestion_GR_horizontal_overflow_image = images[7]
    congestion_GR_vertical_overflow_image = images[8]
    
    image_height, image_width = macro_image.shape
    total_image_area = image_width * image_height
    
    # Convert macro image to [0, 255]
    macro_image = np.uint8(macro_image * 255)
    _, binary_image = cv2.threshold(macro_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_macros = len(contours)
    
    # Compute gradients for congestion images
    grad_x = np.zeros((image_height, image_width))
    grad_y = np.zeros((image_height, image_width))
    
    for img in [congestion_eGR_horizontal_overflow_image, congestion_eGR_vertical_overflow_image,
                congestion_GR_horizontal_overflow_image, congestion_GR_vertical_overflow_image]:
        grad_x += cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        grad_y += cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    
    # Magnitude of gradient
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Calculate average gradient
    avg_gradient = np.mean(grad_magnitude)

    # Convert to um (micrometers), considering pixel size
    avg_gradient_um = avg_gradient * tiles_size
    
    return {"congestion_gradient": avg_gradient_um}

def cell_density_anisotropy(images):
    tiles_size = 2.25
    macro_image = images[0]
    cell_density_image = images[1]

    image_height, image_width = macro_image.shape
    total_image_area = image_width * image_height * (tiles_size ** 2)

    # Convert macro image to [0-255]
    macro_image = np.uint8(macro_image * 255)
    _, binary_image = cv2.threshold(macro_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_macros = len(contours)

    # Compute gradients in the x and y directions
    grad_x = cv2.Sobel(cell_density_image, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(cell_density_image, cv2.CV_64F, 0, 1, ksize=5)
    
    # Compute the magnitude of the gradient
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Calculate anisotropy as the ratio of max gradient to mean gradient
    max_grad = np.max(grad_magnitude)
    mean_grad = np.mean(grad_magnitude)
    
    anisotropy = max_grad / mean_grad if mean_grad != 0 else 0

    return {"cell_density_anisotropy": anisotropy}

def mean_eGR_local_variability(images):
    tiles_size = 2.25
    macro_image = images[0]
    cell_density_image = images[1]
    rudy_long_image = images[2]
    rudy_short_image = images[3]
    rudy_pin_long_image = images[4]
    congestion_eGR_horizontal_overflow_image = images[5]
    congestion_eGR_vertical_overflow_image = images[6]
    congestion_GR_horizontal_overflow_image = images[7]
    congestion_GR_vertical_overflow_image = images[8]
    
    image_height, image_width = macro_image.shape
    total_image_area = image_width * image_height
    
    macro_image = np.uint8(macro_image * 255)
    _, binary_image = cv2.threshold(macro_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_macros = len(contours)

    # Compute the mean congestion overflow for each local region
    window_size = int(10 / tiles_size)  # 10 um regions
    mean_congestion_overflow = []
    
    for y in range(0, image_height, window_size):
        for x in range(0, image_width, window_size):
            # Ensure the window doesn't go out of bounds
            y_end = min(y + window_size, image_height)
            x_end = min(x + window_size, image_width)
            
            # Extract the local region from both horizontal and vertical overflow images
            region_horizontal = congestion_eGR_horizontal_overflow_image[y:y_end, x:x_end]
            region_vertical = congestion_eGR_vertical_overflow_image[y:y_end, x:x_end]
            
            # Calculate the mean overflow for the region
            mean_horizontal = np.mean(region_horizontal)
            mean_vertical = np.mean(region_vertical)
            
            # Average of horizontal and vertical mean overflows
            mean_overflow = (mean_horizontal + mean_vertical) / 2
            mean_congestion_overflow.append(mean_overflow)
    
    # Calculate the variance of the mean congestion overflows across regions
    variability = np.var(mean_congestion_overflow)

    return {"mean_eGR_local_variability": variability}

def diagonal_cell_density_gradient(images):
    tiles_size = 2.25
    macro_image = images[0]
    cell_density_image = images[1]
    
    # Convert the macro image to an appropriate scale
    macro_image = np.uint8(macro_image * 255)
    
    image_height, image_width = cell_density_image.shape
    total_image_area = image_width * image_height
    
    # Calculate the gradient along the diagonals
    # Diagonal gradients: top-left to bottom-right and top-right to bottom-left
    diag1_gradient = np.gradient(cell_density_image, axis=0) + np.gradient(cell_density_image, axis=1)
    diag2_gradient = np.gradient(cell_density_image, axis=0) - np.gradient(cell_density_image, axis=1)
    
    # Calculate the average gradient magnitude along the diagonals
    diag1_magnitude = np.sqrt(diag1_gradient**2)
    diag2_magnitude = np.sqrt(diag2_gradient**2)
    
    # Average the diagonal gradients
    avg_diagonal_gradient = (np.sum(diag1_magnitude) + np.sum(diag2_magnitude)) / (2 * total_image_area)
    
    # Convert from pixels to um (micro-meters) (1 pixel = 2.25um)
    avg_diagonal_gradient_um = avg_diagonal_gradient * tiles_size
    
    return {"diagonal_cell_density_gradient": avg_diagonal_gradient_um}


def mean_cell_density_fluctuation(images):
    tiles_size = 2.25
    macro_image = images[0]
    cell_density_image = images[1]
    rudy_long_image = images[2]
    rudy_short_image = images[3]
    rudy_pin_long_image = images[4]
    congestion_eGR_horizontal_overflow_image = images[5]
    congestion_eGR_vertical_overflow_image = images[6]
    congestion_GR_horizontal_overflow_image = images[7]
    congestion_GR_vertical_overflow_image = images[8]
    
    image_height, image_width = macro_image.shape
    
    macro_image = np.uint8(macro_image * 255)
    _, binary_image = cv2.threshold(macro_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Normalize cell density image to [0, 1]
    cell_density_image = np.clip(cell_density_image, 0, 1)

    # Calculate fluctuation
    cell_density_gradient_x = cv2.Sobel(cell_density_image, cv2.CV_64F, 1, 0, ksize=3)
    cell_density_gradient_y = cv2.Sobel(cell_density_image, cv2.CV_64F, 0, 1, ksize=3)
    cell_density_fluctuation = np.sqrt(cell_density_gradient_x**2 + cell_density_gradient_y**2)

    # Calculate mean fluctuation
    mean_fluctuation = np.mean(cell_density_fluctuation)

    # Scale to micrometers
    fluctuation_um = mean_fluctuation * tiles_size

    return {"mean_cell_density_fluctuation": fluctuation_um}


def macro_transition_band(images):
    tiles_size = 2.25
    macro_image = images[0]
    cell_density_image = images[1]
    rudy_long_image = images[2]
    rudy_short_image = images[3]
    rudy_pin_long_image = images[4]
    congestion_eGR_horizontal_overflow_image = images[5]
    congestion_eGR_vertical_overflow_image = images[6]
    congestion_GR_horizontal_overflow_image = images[7]
    congestion_GR_vertical_overflow_image = images[8]

    image_height, image_width = macro_image.shape
    total_image_area = image_width * image_height

    macro_image = np.uint8(macro_image * 255)
    _, binary_image = cv2.threshold(macro_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    macro_transition_area = np.zeros_like(macro_image)

    # Extract transition bands around macros
    for contour in contours:
        cv2.drawContours(macro_transition_area, [contour], -1, 255, thickness=5)

    transition_band_mask = macro_transition_area > 0

    merged_features = np.maximum.reduce([
        cell_density_image,
        rudy_long_image,
        rudy_short_image,
        rudy_pin_long_image,
        congestion_eGR_horizontal_overflow_image,
        congestion_eGR_vertical_overflow_image,
        congestion_GR_horizontal_overflow_image,
        congestion_GR_vertical_overflow_image
    ])

    # Identify rapid change areas
    feature_difference = cv2.Laplacian(merged_features, cv2.CV_64F)
    feature_difference = np.abs(feature_difference)

    macro_transition_band_area = np.sum(feature_difference[transition_band_mask])

    # Convert pixel area to physical area (um^2)
    physical_macro_transition_band_area = macro_transition_band_area * (tiles_size ** 2)

    return {"macro_transition_band": physical_macro_transition_band_area}



def cell_density_skewness(images):
    tiles_size = 2.25
    macro_image = images[0]
    cell_density_image = images[1]
    
    image_height, image_width = macro_image.shape
    total_image_area = image_width * image_height
    
    macro_image = np.uint8(macro_image * 255)
    _, binary_image = cv2.threshold(macro_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_macros = len(contours)
    
    # Calculate skewness of cell density
    cell_density_values = cell_density_image.flatten()
    cell_density_skewness = skew(cell_density_values)
    
    return {"cell_density_skewness": cell_density_skewness}

def cell_density_skewness_gradient(images):
    tiles_size = 2.25
    macro_image = images[0]
    cell_density_image = images[1]
    rudy_long_image = images[2]
    rudy_short_image = images[3]
    rudy_pin_long_image = images[4]
    congestion_eGR_horizontal_overflow_image = images[5]
    congestion_eGR_vertical_overflow_image = images[6]
    congestion_GR_horizontal_overflow_image = images[7]
    congestion_GR_vertical_overflow_image = images[8]
    
    image_height, image_width = macro_image.shape
    total_image_area = image_width * image_height
    
    macro_image = np.uint8(macro_image * 255)
    _, binary_image = cv2.threshold(macro_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_macros = len(contours)
    
    # Normalize cell_density_image to [0, 255] for visualization
    cell_density_image = (cell_density_image * 255).astype(np.uint8)

    # Calculate gradients using Sobel operator
    grad_x = cv2.Sobel(cell_density_image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(cell_density_image, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate magnitude of gradients
    magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Calculate skewness of the gradient magnitudes
    skewness_gradient = skew(magnitude.flatten())

    # Convert skewness value to um (consider image resolution)
    feature_value = skewness_gradient * (tiles_size ** 2)
    
    return {"cell_density_skewness_gradient": feature_value}

def macro_interaction_perimeter(images):
    tile_size = 2.25
    macro_image = images[0]
    
    # Convert macro image to [0-255] scale
    macro_image = np.uint8(macro_image * 255)
    
    # Threshold to create a binary image
    _, binary_image = cv2.threshold(macro_image, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calculate total interaction perimeter
    interaction_perimeter = 0
    for contour in contours:
        # Approximate the contour to avoid small variations
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Calculate perimeter of the approximated contour
        perimeter = cv2.arcLength(approx, True)
        interaction_perimeter += perimeter

    # Convert perimeter from pixels to micrometers
    interaction_perimeter_um = interaction_perimeter * tile_size
    
    return {"macro_interaction_perimeter": interaction_perimeter_um}

def cell_density_fluctuation_balance(images):
    tiles_size = 2.25
    macro_image = images[0]
    cell_density_image = images[1]
    
    # Convert the macro image
    macro_image = np.uint8(macro_image * 255)
    
    # Threshold for macro regions
    _, binary_image = cv2.threshold(macro_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_macros = len(contours)
    
    # Calculate standard deviation in cell density image
    cell_density_std = np.std(cell_density_image)

    # Identify regions of increasing and decreasing fluctuations
    high_fluctuation_mask = cell_density_image > np.mean(cell_density_image) + cell_density_std
    low_fluctuation_mask = cell_density_image < np.mean(cell_density_image) - cell_density_std
    
    # Calculate area for increasing and decreasing fluctuations
    increasing_fluctuation_area = np.sum(high_fluctuation_mask) * (tiles_size ** 2)
    decreasing_fluctuation_area = np.sum(low_fluctuation_mask) * (tiles_size ** 2)
    
    # Calculate balance
    fluctuation_balance = increasing_fluctuation_area / (decreasing_fluctuation_area + 1e-6)  # avoid division by zero

    return {"cell_density_fluctuation_balance": fluctuation_balance}

def congestion_pressure_fluctuation(images):
    tiles_size = 2.25
    macro_image = images[0]
    cell_density_image = images[1]
    rudy_long_image = images[2]
    rudy_short_image = images[3]
    rudy_pin_long_image = images[4]
    congestion_eGR_horizontal_overflow_image = images[5]
    congestion_eGR_vertical_overflow_image = images[6]
    congestion_GR_horizontal_overflow_image = images[7]
    congestion_GR_vertical_overflow_image = images[8]

    image_height, image_width = macro_image.shape
    total_image_area = image_width * image_height

    # Convert macro_image to [0-255]
    macro_image = np.uint8(macro_image * 255)
    _, binary_image = cv2.threshold(macro_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_macros = len(contours)

    # Calculate congestion fluctuation
    congestion_images = [
        congestion_eGR_horizontal_overflow_image,
        congestion_eGR_vertical_overflow_image,
        congestion_GR_horizontal_overflow_image,
        congestion_GR_vertical_overflow_image,
    ]

    fluctuation_totals = []

    for cong_img in congestion_images:
        mean, std_dev = cv2.meanStdDev(cong_img)
        fluctuation_totals.append(std_dev[0][0])

    average_fluctuation = np.mean(fluctuation_totals)
    
    # Calculate feature in um
    average_fluctuation_um = average_fluctuation * (tiles_size * image_width)

    feature_value = average_fluctuation_um

    return {"congestion_pressure_fluctuation": feature_value}

def congestion_variability_throughout_hierarchy(images):
    tiles_size = 2.25
    macro_image = images[0]
    cell_density_image = images[1]
    rudy_long_image = images[2]
    rudy_short_image = images[3]
    rudy_pin_long_image = images[4]
    congestion_eGR_horizontal_overflow_image = images[5]
    congestion_eGR_vertical_overflow_image = images[6]
    congestion_GR_horizontal_overflow_image = images[7]
    congestion_GR_vertical_overflow_image = images[8]
    
    image_height, image_width = macro_image.shape
    total_image_area = image_width * image_height
    
    # Convert macro image to [0, 255] and binarize
    macro_image = np.uint8(macro_image * 255)
    _, binary_image = cv2.threshold(macro_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_macros = len(contours)
    
    # Compute congestion variability across hierarchical images
    congestion_images = [
        congestion_eGR_horizontal_overflow_image, 
        congestion_eGR_vertical_overflow_image,
        congestion_GR_horizontal_overflow_image, 
        congestion_GR_vertical_overflow_image
    ]
    
    congestion_variability = []
    
    for img in congestion_images:
        non_zero_congestion = img[img > 0]  # Get non-zero congestion values
        if len(non_zero_congestion) > 0:
            variability = np.std(non_zero_congestion) / np.mean(non_zero_congestion)
            congestion_variability.append(variability)
    
    avg_congestion_variability = np.mean(congestion_variability) if congestion_variability else 0
    
    # The feature is the average variability converted to the area unit
    feature_value = avg_congestion_variability * tiles_size**2
    
    return {"congestion_variability_throughout_hierarchy": feature_value}

def congestion_transition_amplitude(images):
    tiles_size = 2.25
    macro_image = images[0]
    cell_density_image = images[1]
    rudy_long_image = images[2]
    rudy_short_image = images[3]
    rudy_pin_long_image = images[4]
    congestion_eGR_horizontal_overflow_image = images[5]
    congestion_eGR_vertical_overflow_image = images[6]
    congestion_GR_horizontal_overflow_image = images[7]
    congestion_GR_vertical_overflow_image = images[8]
    
    # Convert macro image to [0, 255]
    macro_image = np.uint8(macro_image * 255)
    
    # Concatenate all congestion images into a single stack
    congestion_images = [
        congestion_eGR_horizontal_overflow_image, congestion_eGR_vertical_overflow_image,
        congestion_GR_horizontal_overflow_image, congestion_GR_vertical_overflow_image
    ]

    # Calculate the gradient (Sobel) to find edges representing congestion transitions
    transition_magnitudes = []
    for img in congestion_images:
        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate the magnitude of the gradient
        magnitude = cv2.magnitude(sobel_x, sobel_y)
        transition_magnitudes.append(np.sum(magnitude))
    
    # Calculate the mean transition amplitude for the entire layout
    avg_transition_amplitude = np.mean(transition_magnitudes) * tiles_size
    
    return {"congestion_transition_amplitude": avg_transition_amplitude}

def cell_density_dipole(images):
    tiles_size = 2.25
    macro_image = images[0]
    cell_density_image = images[1]

    # Convert macro image to [0-255]
    macro_image = np.uint8(macro_image * 255)

    # Threshold macro image to find macros
    _, binary_image = cv2.threshold(macro_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_macros = len(contours)

    # Calculate the standard deviation of cell densities
    cell_density_std = np.std(cell_density_image)

    # Calculate dipole effect by finding areas with significantly different cell density
    high_density_areas = np.where(cell_density_image > (0.5 + cell_density_std), 1, 0)
    low_density_areas = np.where(cell_density_image < (0.5 - cell_density_std), 1, 0)
    dipole_effect = np.sum(high_density_areas) * tiles_size ** 2 + np.sum(low_density_areas) * tiles_size ** 2

    feature_value = dipole_effect

    return {"cell_density_dipole": feature_value}

def mean_eGR_local_adjacent_cohesion(images):
    tiles_size = 2.25
    macro_image = images[0]
    cell_density_image = images[1]
    rudy_long_image = images[2]
    rudy_short_image = images[3]
    rudy_pin_long_image = images[4]
    congestion_eGR_horizontal_overflow_image = images[5]
    congestion_eGR_vertical_overflow_image = images[6]
    congestion_GR_horizontal_overflow_image = images[7]
    congestion_GR_vertical_overflow_image = images[8]
    
    image_height, image_width = macro_image.shape
    total_image_area = image_width * image_height
    
    macro_image = np.uint8(macro_image * 255)
    _, binary_image = cv2.threshold(macro_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_macros = len(contours)
    
    # Compute mean eGR local adjacent cohesion
    # Average gradients in congestion overflow images
    sobelx_hor = cv2.Sobel(congestion_eGR_horizontal_overflow_image, cv2.CV_64F, 1, 0, ksize=3)
    sobely_hor = cv2.Sobel(congestion_eGR_horizontal_overflow_image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude_hor = np.sqrt(sobelx_hor**2 + sobely_hor**2)
    
    sobelx_ver = cv2.Sobel(congestion_eGR_vertical_overflow_image, cv2.CV_64F, 1, 0, ksize=3)
    sobely_ver = cv2.Sobel(congestion_eGR_vertical_overflow_image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude_ver = np.sqrt(sobelx_ver**2 + sobely_ver**2)

    mean_gradient_magnitude_hor = np.mean(gradient_magnitude_hor)
    mean_gradient_magnitude_ver = np.mean(gradient_magnitude_ver)
    
    mean_eGR_local_adjacent_cohesion = (mean_gradient_magnitude_hor + mean_gradient_magnitude_ver) / 2
    
    return {"mean_eGR_local_adjacent_cohesion": mean_eGR_local_adjacent_cohesion}


feat_func_list = [max_congestion_ripple,
 macro_interference_zone,
 macro_compactness_index,
 cell_density_variance_gradient,
 mean_macro_proximity,
 congestion_gradient,
 cell_density_anisotropy,
 mean_eGR_local_variability,
 diagonal_cell_density_gradient,
 mean_cell_density_fluctuation,
 macro_transition_band,
 cell_density_skewness,
 cell_density_skewness_gradient,
 macro_interaction_perimeter,
 cell_density_fluctuation_balance,
 congestion_pressure_fluctuation,
 congestion_variability_throughout_hierarchy,
 congestion_transition_amplitude,
 cell_density_dipole,
 mean_eGR_local_adjacent_cohesion]