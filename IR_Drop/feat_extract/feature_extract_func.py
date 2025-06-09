import numpy as np
import cv2
from scipy.stats import kurtosis
from scipy.stats import skew

feat_pool = {'horizontal_power_distribution_symmetry': 'symmetry level of power distribution along the horizontal axis across the layout',
 'mean_power_sca': 'the average of the toggle rate scaled power in the layout',
 'heat_intensity_correlation': 'correlation between power density and expected thermal hotspots',
 'central_power_saturation': 'the saturation level of power distribution concentrated at the central region of the layout',
 'vertical_power_distribution_symmetry': 'symmetry level of power distribution along the vertical axis across the layout',
 'proximity_power_pattern_asymmetry': 'the asymmetry in power distribution patterns between neighboring regions indicating potential IR Drop irregularities',
 'macro_power_proximity': 'the influence of power distribution in proximity to macros',
 'mean_power_density_deviation': 'the average deviation of power density from the mean across the layout regions',
 'edge_power_intensity': 'the intensity of power distribution along the edges of the layout',
 'power_sink_effect': 'the effect of specific regions acting as sinks, causing power absorption',
 'mean_power_all': 'the average of power_all = power_i + power_s + power_sca in the layout',
 'mean_power_i': 'the average of the internal power in the layout',
 'power_balance_ratio': 'the balance ratio of power distribution between high and low intensity areas',
 'power_gradient_variation': 'the variation in power gradient magnitude across the layout',
 'localized_coupling_variability': 'the variation in coupling strength between neighboring power points across different regions of the layout',
 'power_intensity_anomaly_detection': 'identification of anomalies in power intensity not adhering to expected patterns',
 'localized_gradient_intensity': 'the intensity of power gradient variations within localized regions of the layout',
 'spatial_correlation_power_i': 'the spatial correlation of internal power fluctuations across the layout',
 'uniformity_index_power_i': 'uniformity index of internal power spread across the entire layout',
 'spatial_density_power_i': 'the spatial density of internal power distribution in the layout'}

def horizontal_power_distribution_symmetry(images):
    # Aggregate power values across all images
    power_i = images[0]
    power_s = images[1]
    power_sca = images[2]
    power_all = images[3]
    power_t_6 = images[4]
    power_t_13 = images[5]
    power_t_19 = images[6]

    # Compute horizontal symmetry for a single image
    def compute_symmetry(image):
        # Flip the image horizontally
        flipped_image = cv2.flip(image, 1)
        
        # Compute symmetry as the sum of absolute differences
        symmetry = np.sum(np.abs(image - flipped_image))
        
        return symmetry
    
    # Compute the average symmetry across all power images
    symmetry_values = [
        compute_symmetry(power_i),
        compute_symmetry(power_s),
        compute_symmetry(power_sca),
        compute_symmetry(power_all),
        compute_symmetry(power_t_6),
        compute_symmetry(power_t_13),
        compute_symmetry(power_t_19),
    ]
    
    # Calculate the overall symmetry measure
    average_symmetry = np.mean(symmetry_values)
    
    return {"horizontal_power_distribution_symmetry": average_symmetry}

def mean_power_sca(images):
    image = images[2]
    total_area = image.shape[0] * image.shape[1]
    m_power_sca = np.mean(image)
    
    return {"mean_power_sca": m_power_sca}

def heat_intensity_correlation(images):
    power_i = images[0]
    power_s = images[1]
    power_sca = images[2]
    power_all = images[3]
    
    # Convert images to NumPy arrays if they are not already
    power_i = np.array(power_i, dtype=np.float32)
    power_s = np.array(power_s, dtype=np.float32)
    power_sca = np.array(power_sca, dtype=np.float32)
    power_all = np.array(power_all, dtype=np.float32)
    
    # Flatten the images for correlation calculation
    power_i_flat = power_i.flatten()
    power_s_flat = power_s.flatten()
    power_sca_flat = power_sca.flatten()
    power_all_flat = power_all.flatten()
    
    # Calculate correlation coefficients
    corr_i = np.corrcoef(power_i_flat, power_all_flat)[0, 1]
    corr_s = np.corrcoef(power_s_flat, power_all_flat)[0, 1]
    corr_sca = np.corrcoef(power_sca_flat, power_all_flat)[0, 1]
    
    # Average of the correlations as a simple metric
    heat_intensity_correlation_value = (corr_i + corr_s + corr_sca) / 3
    
    return {"heat_intensity_correlation": heat_intensity_correlation_value}

def central_power_saturation(images):
    # Extracting image data
    power_i = images[0]
    power_s = images[1]
    power_sca = images[2]
    power_all = images[3]
    power_t_6 = images[4]
    power_t_13 = images[5]
    power_t_19 = images[6]

    # Combining the power images
    combined_power = (power_i + power_s + power_sca + power_all + power_t_6 + power_t_13 + power_t_19) / 7

    # Get the dimensions of the image
    height, width = combined_power.shape

    # Define the central region as the middle 50% of the image
    central_region = combined_power[int(0.25 * height):int(0.75 * height), int(0.25 * width):int(0.75 * width)]

    # Calculate the mean power in the central region
    central_power_mean = np.mean(central_region)

    # Calculate the overall mean power in the whole image
    overall_power_mean = np.mean(combined_power)

    # Calculate the central power saturation as a ratio
    central_power_saturation = central_power_mean / overall_power_mean if overall_power_mean != 0 else 0

    return {"central_power_saturation": central_power_saturation}

def vertical_power_distribution_symmetry(images):
    # Extract images
    power_i = images[0]
    power_s = images[1]
    power_sca = images[2]
    power_all = images[3]
    power_t_6 = images[4]
    power_t_13 = images[5]
    power_t_19 = images[6]

    # Select the image to analyze; using power_all as an example
    image = power_all

    # Get the dimensions of the image
    height, width = image.shape

    # Split the image into left and right halves
    left_half = image[:, :width // 2]
    right_half = image[:, width // 2:]

    # Flip the right half horizontally
    right_half_flipped = cv2.flip(right_half, 1)

    # Calculate the absolute difference between the halves
    difference = np.abs(left_half - right_half_flipped)

    # Calculate symmetry score
    symmetry_score = 1 - np.sum(difference) / np.sum(image)

    # Normalize to make sure the result is between 0 and 1
    symmetry_score = np.clip(symmetry_score, 0, 1)

    return {"vertical_power_distribution_symmetry": symmetry_score}

def proximity_power_pattern_asymmetry(images):
    power_i = images[0]
    power_s = images[1]
    power_sca = images[2]
    power_all = images[3]
    power_t_6 = images[4]
    power_t_13 = images[5]
    power_t_19 = images[6]
    
    # Function to calculate local asymmetry in a power map
    def calculate_asymmetry(image):
        # Compute Sobel gradients to find edges
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        # Compute the gradient magnitude
        mag = cv2.magnitude(grad_x, grad_y)
        
        # Normalize the gradient magnitude
        mag /= np.max(mag)
        
        # Calculate asymmetry by checking variance of gradient magnitudes 
        # in local neighborhoods
        asymmetry = cv2.Laplacian(mag, cv2.CV_64F)
        
        # Calculate the overall asymmetry as the standard deviation of 
        # the asymmetry map
        asymmetry_value = np.mean(np.abs(asymmetry))
        
        return asymmetry_value

    # Calculate asymmetry for each power map
    asymmetry_i = calculate_asymmetry(power_i)
    asymmetry_s = calculate_asymmetry(power_s)
    asymmetry_sca = calculate_asymmetry(power_sca)
    asymmetry_all = calculate_asymmetry(power_all)
    asymmetry_t_6 = calculate_asymmetry(power_t_6)
    asymmetry_t_13 = calculate_asymmetry(power_t_13)
    asymmetry_t_19 = calculate_asymmetry(power_t_19)

    # Compute a combined asymmetry value
    feature_value = (asymmetry_i + asymmetry_s + asymmetry_sca + asymmetry_all +
                     asymmetry_t_6 + asymmetry_t_13 + asymmetry_t_19) / 7

    return {"proximity_power_pattern_asymmetry": feature_value}

def macro_power_proximity(images):
    # Extract the relevant images
    power_i = images[0]
    power_s = images[1]
    power_sca = images[2]
    power_all = images[3]
    power_t_6 = images[4]
    power_t_13 = images[5]
    power_t_19 = images[6]
    
    # Calculate gradients to understand power distribution around macros
    grad_x = cv2.Sobel(power_all, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(power_all, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute the magnitude of the gradient
    gradient_magnitude = cv2.magnitude(grad_x, grad_y)
    
    # Normalize the gradient magnitude to analyze relative distribution
    normalized_magnitude = cv2.normalize(gradient_magnitude, None, 0, 1, cv2.NORM_MINMAX)
    
    # Calculate the mean proximity influence of power distribution
    # This metric gives a sense of variance or spread of power around macros
    macro_power_proximity_value = np.mean(normalized_magnitude)
    
    # Return the feature as a dictionary
    return {"macro_power_proximity": macro_power_proximity_value}

def mean_power_density_deviation(images):
    power_i = images[0]
    power_s = images[1]
    power_sca = images[2]
    power_all = images[3]
    power_t_6 = images[4]
    power_t_13 = images[5]
    power_t_19 = images[6]

    # Calculate the mean power for each image
    mean_power_i = np.mean(power_i)
    mean_power_s = np.mean(power_s)
    mean_power_sca = np.mean(power_sca)
    mean_power_all = np.mean(power_all)
    mean_power_t_6 = np.mean(power_t_6)
    mean_power_t_13 = np.mean(power_t_13)
    mean_power_t_19 = np.mean(power_t_19)
    
    # Calculate deviations from the mean
    deviation_i = np.abs(power_i - mean_power_i)
    deviation_s = np.abs(power_s - mean_power_s)
    deviation_sca = np.abs(power_sca - mean_power_sca)
    deviation_all = np.abs(power_all - mean_power_all)
    deviation_t_6 = np.abs(power_t_6 - mean_power_t_6)
    deviation_t_13 = np.abs(power_t_13 - mean_power_t_13)
    deviation_t_19 = np.abs(power_t_19 - mean_power_t_19)

    # Average the deviations
    average_deviation = (
        np.mean(deviation_i) + 
        np.mean(deviation_s) + 
        np.mean(deviation_sca) + 
        np.mean(deviation_all) + 
        np.mean(deviation_t_6) + 
        np.mean(deviation_t_13) + 
        np.mean(deviation_t_19)
    ) / 7

    return {"mean_power_density_deviation": average_deviation}


def edge_power_intensity(images):
    power_i = images[0]
    power_s = images[1]
    power_sca = images[2]
    power_all = images[3]
    power_t_6 = images[4]
    power_t_13 = images[5]
    power_t_19 = images[6]

    # Convert images to the format required by OpenCV (0-255 range)
    power_all_scaled = (power_all * 255).astype(np.uint8)

    # Detect edges using the Canny edge detector
    edges = cv2.Canny(power_all_scaled, threshold1=30, threshold2=100)

    # Calculate the intensity of the power along the edges
    edge_intensity_sum = np.sum(power_all[edges > 0])

    # Normalize the intensity by the number of edge pixels
    num_edge_pixels = np.sum(edges > 0)
    if num_edge_pixels > 0:
        edge_power_intensity = edge_intensity_sum / num_edge_pixels
    else:
        edge_power_intensity = 0

    return {"edge_power_intensity": edge_power_intensity}

def power_sink_effect(images):
    power_i = images[0]
    power_s = images[1]
    power_sca = images[2]
    power_all = images[3]
    power_t_6 = images[4]
    power_t_13 = images[5]
    power_t_19 = images[6]

    # Step 1: Calculate the gradient magnitude of the power_all image to detect regions with high power changes
    grad_x = cv2.Sobel(power_all, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(power_all, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = cv2.magnitude(grad_x, grad_y)

    # Step 2: Assess the variations in power over time to identify sink regions
    power_difference_6_13 = power_t_13 - power_t_6
    power_difference_13_19 = power_t_19 - power_t_13

    # Step 3: Combine the gradient and power differences to highlight potential sink areas
    combined_effect = gradient_magnitude + np.abs(power_difference_6_13) + np.abs(power_difference_13_19)

    # Step 4: Normalize the result to keep values between 0 and 1
    power_sink_effect_value = cv2.normalize(combined_effect, None, 0, 1, cv2.NORM_MINMAX)

    return {"power_sink_effect": np.mean(power_sink_effect_value)}

def mean_power_all(images):
    image = images[3]
    total_area = image.shape[0] * image.shape[1]
    m_power_all = np.mean(image)
    
    return {"mean_power_all": m_power_all}


def mean_power_i(images):
    image = images[0]
    total_area = image.shape[0] * image.shape[1]
    m_power_i = np.mean(image)
    
    return {"mean_power_i": m_power_i}

def power_balance_ratio(images):
    # Extract the images
    power_i = images[0]
    power_s = images[1]
    power_sca = images[2]
    power_all = images[3]
    power_t_6 = images[4]
    power_t_13 = images[5]
    power_t_19 = images[6]

    # Use the power_all for calculation as it's the sum of other powers
    img = power_all

    # Normalize the image to [0, 255] for thresholding
    img_scaled = np.uint8(img * 255)

    # Determine threshold using Otsu's method
    _, thresholded = cv2.threshold(img_scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Calculate the sum of power for high and low intensity areas
    high_intensity_sum = np.sum(img[thresholded == 255])
    low_intensity_sum = np.sum(img[thresholded == 0])

    # Handle division by zero in case all pixels are in one category
    if high_intensity_sum + low_intensity_sum == 0:
        balance_ratio = 0
    else:
        balance_ratio = high_intensity_sum / (high_intensity_sum + low_intensity_sum)

    return {"power_balance_ratio": balance_ratio}

def power_gradient_variation(images):
    power_i = images[0]
    power_s = images[1]
    power_sca = images[2]
    power_all = images[3]
    power_t_6 = images[4]
    power_t_13 = images[5]
    power_t_19 = images[6]

    # Function to compute the gradient magnitude
    def gradient_magnitude(image):
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(sobelx, sobely)
        return magnitude

    # Compute the gradient magnitude for each power image
    gradients = [gradient_magnitude(img) for img in images]

    # Calculate the variation (standard deviation) in gradient magnitudes
    gradient_stack = np.stack(gradients, axis=0)
    gradient_variation = np.std(gradient_stack, axis=0)
    
    # Compute the overall variation value as mean of gradient variation
    feature_value = np.mean(gradient_variation)
    
    return {"power_gradient_variation": feature_value}

def localized_coupling_variability(images):
    power_i = images[0]
    power_s = images[1]
    power_sca = images[2]
    power_all = images[3]
    power_t_6 = images[4]
    power_t_13 = images[5]
    power_t_19 = images[6]

    def calculate_variability(image):
        # Calculate gradients in x and y directions
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate the magnitude of the gradients
        magnitude = cv2.magnitude(grad_x, grad_y)
        
        # Return the average variability across the image
        variability = np.mean(magnitude)
        return variability

    # Calculate the variability for each power image
    variability_i = calculate_variability(power_i)
    variability_s = calculate_variability(power_s)
    variability_sca = calculate_variability(power_sca)
    variability_all = calculate_variability(power_all)

    # Aggregate the variability values
    localized_variability = (variability_i + variability_s + variability_sca + variability_all) / 4

    return {"localized_coupling_variability": localized_variability}

def power_intensity_anomaly_detection(images):
    power_i = images[0]
    power_s = images[1]
    power_sca = images[2]
    power_all = images[3]
    power_t_6 = images[4]
    power_t_13 = images[5]
    power_t_19 = images[6]

    # Calculate differences between power_all and individual components to detect anomalies
    diff_i = cv2.absdiff(power_all, power_i)
    diff_s = cv2.absdiff(power_all, power_s)
    diff_sca = cv2.absdiff(power_all, power_sca)
    
    # Calculate temporal differences to detect sudden changes
    temporal_diff_6_13 = cv2.absdiff(power_t_6, power_t_13)
    temporal_diff_13_19 = cv2.absdiff(power_t_13, power_t_19)

    # Aggregate anomaly indicators
    anomaly_indicator = (diff_i + diff_s + diff_sca +
                         temporal_diff_6_13 + temporal_diff_13_19)

    # Threshold the anomaly indicator to create a binary anomaly map
    _, anomaly_map = cv2.threshold(anomaly_indicator, 0.1, 1, cv2.THRESH_BINARY)

    # Calculate the feature value as the sum of anomaly map values
    feature_value = np.sum(anomaly_map)

    return {"power_intensity_anomaly_detection": feature_value}


def localized_gradient_intensity(images):
    # Unpack images
    power_i = images[0]
    power_s = images[1]
    power_sca = images[2]
    power_all = images[3]
    power_t_6 = images[4]
    power_t_13 = images[5]
    power_t_19 = images[6]

    # Initialize gradient intensity accumulator
    gradient_intensity_sum = 0

    # Iterate through each image to calculate gradient
    for image in [power_i, power_s, power_sca, power_all, power_t_6, power_t_13, power_t_19]:
        # Generate gradient maps using Sobel operator
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate the gradient magnitude
        grad_magnitude = cv2.magnitude(grad_x, grad_y)
        
        # Sum gradient magnitudes over the image
        gradient_intensity_sum += np.sum(grad_magnitude)

    # Calculate the average gradient intensity for all images
    num_images = len(images)
    average_gradient_intensity = gradient_intensity_sum / num_images

    return {"localized_gradient_intensity": average_gradient_intensity}

def spatial_correlation_power_i(images):
    power_i = images[0]
    
    def calculate_spatial_correlation(image):
        # Compute the 2D Fourier Transform of the image
        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)
        
        # Compute the magnitude spectrum
        magnitude_spectrum = np.abs(f_shift)
        
        # Calculate the autocorrelation using inverse FFT of magnitude squared
        autocorrelation = np.fft.ifft2(magnitude_spectrum**2)
        
        # Shift zero frequency component back to center
        autocorrelation = np.fft.fftshift(autocorrelation)
        
        # Normalize the result
        autocorrelation = np.abs(autocorrelation)
        autocorrelation /= autocorrelation.max()
        
        # Calculate mean spatial correlation
        spatial_correlation = np.mean(autocorrelation)
        
        return spatial_correlation
    
    # Calculate the spatial correlation for internal power fluctuations
    spatial_correlation_value = calculate_spatial_correlation(power_i)
    
    return {"spatial_correlation_power_i": spatial_correlation_value}

def uniformity_index_power_i(images):
    power_i = images[0]
    
    # Convert the image to a NumPy array if it's not already
    power_i_array = np.array(power_i, dtype=np.float32)
    
    # Calculate the mean and standard deviation of the power_i image
    mean_power_i = np.mean(power_i_array)
    std_power_i = np.std(power_i_array)
    
    # Calculate the uniformity index
    if mean_power_i != 0:
        uniformity_index = 1 / (std_power_i / mean_power_i)
    else:
        uniformity_index = 0  # Handle division by zero for a perfectly uniform image

    return {"uniformity_index_power_i": uniformity_index}


def spatial_density_power_i(images):
    power_i = images[0]
    
    # Normalize image to the range 0-255 for OpenCV processing
    power_i = np.uint8(power_i * 255)
    
    # Threshold the image to create a binary map of significant power regions
    _, binary_map = cv2.threshold(power_i, 1, 255, cv2.THRESH_BINARY)
    
    # Calculate the spatial density as the ratio of non-zero pixels to total pixels
    non_zero_count = np.count_nonzero(binary_map)
    total_pixels = binary_map.size
    spatial_density = non_zero_count / total_pixels
    
    return {"spatial_density_power_i": spatial_density}


feat_func_list = [horizontal_power_distribution_symmetry,
 mean_power_sca,
 heat_intensity_correlation,
 central_power_saturation,
 vertical_power_distribution_symmetry,
 proximity_power_pattern_asymmetry,
 macro_power_proximity,
 mean_power_density_deviation,
 edge_power_intensity,
 power_sink_effect,
 mean_power_all,
 mean_power_i,
 power_balance_ratio,
 power_gradient_variation,
 localized_coupling_variability,
 power_intensity_anomaly_detection,
 localized_gradient_intensity,
 spatial_correlation_power_i,
 uniformity_index_power_i,
 spatial_density_power_i]