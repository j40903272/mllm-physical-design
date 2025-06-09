import cv2
import numpy as np
import re

feat_pool = {'total_wirelength': 'The total length of all routed wires in the design, which is a key indicator of routing complexity and congestion.',
 'number_vias': 'The total number of vias in the layout, which represent connections between different metal layers.',
 'number_of_multi_cut_vias': 'The number of vias that use multiple cuts for enhanced reliability and lower resistance.',
 'number_of_single_cut_vias': 'The number of vias that use a single cut, which are more susceptible to manufacturing defects but occupy less space.',
 'max_overcon': 'The maximum congestion value observed in any global routing cell, indicating the worst-case routing bottleneck.',
 'total_overcon': 'The sum of all routing congestion values across the entire layout, representing overall congestion severity.',
 'worst_layer_gcell_overcon_rate': 'The highest over-congestion rate observed in any layer for a global routing cell, indicating the most problematic metal layer in terms of congestion.',
 'hard_to_access_pins_ratio': 'The ratio of pins that are difficult to access during routing, usually due to blockages, which can limit routing options and cause congestion.',
 'instance_blockages_count': 'The total number of instance blockages, providing a measure of how much of the design is obstructed by existing instances, impacting routing options.',
 'early_gr_overflow_percentage': 'The percentage of routing overflow observed during early global routing, providing a preliminary measure of routing congestion.',
 'horizontal_overflow_percentage': 'The percentage of congestion due to horizontal routing overflows, providing insights into layers where horizontal routing capacity might be insufficient.',
 'initial_placement_efficiency': 'A measure of how efficiently the initial placement of macros, standard cells, and other design elements is conducted, impacting the ease of routing and minimizing early congestion points.',
 'congestion_prediction_accuracy': 'The accuracy of congestion predictions made during routing planning phases compared to actual outcomes, providing a measure of predictive effectiveness.',
 'area_based_congestion_density': 'A metric assessing the congestion density relative to specific design areas, allowing for targeted congestion alleviation strategies focused on high-density regions.',
 'multi_layer_pin_access_variability': 'The variability in pin access difficulty across multiple metal layers, indicating areas where certain layers might present more challenges to effective routing due to blockages or limited access paths, impacting the generation of a comprehensive congestion map.',
 'average_layer_congestion': 'The average congestion value across all layers in the design, providing an overall sense of how balanced the congestion is distributed among different layers.',
 'pin_density_variance_map': 'A mapping of the variance in pin density across different regions of the design, providing insights into potential areas where high pin density might contribute to routing difficulties and congestion.',
 'non_default_routing_rule_usage': 'Tracks the frequency and distribution of non-default routing rules applied, as these indicate areas where special routing considerations are needed to reduce congestion.',
 'crosstalk_sensitive_zones': 'Regions of the design where potential crosstalk is amplified due to adjacent routed wires, impacting signal integrity and possibly creating indirect congestion through rerouting efforts.',
 'inter_macro_channel_congestion': 'Specifically targets congestion in channels between macros, where routing requirements often peak due to limited passageways and over-utilized resources.'}

def total_wirelength(logging_file_string):
    matches = re.findall(r"Total wire length =\s*([\d.]+)", logging_file_string)
    if matches:
        ans = float(matches[-1])
    else:
        ans = 0
        
    return {"total_wirelength": ans}
    
def number_vias(logging_file_string):
    matches = re.findall(r"Total number of vias =\s*([\d.]+)", logging_file_string)
    if matches:
        ans = float(matches[-1])
    else:
        ans = 0
        
    return {"number_vias": ans}
    
def number_of_multi_cut_vias(logging_file_string):
    matches = re.findall(r"Total number of multi-cut vias =\s*([\d.]+)", logging_file_string)
    if matches:
        ans = float(matches[-1])
    else:
        ans = 0
        
    return {"number_of_multi_cut_vias": ans}
    
def number_of_single_cut_vias(logging_file_string):
    matches = re.findall(r"Total number of single cut vias =\s*([\d.]+)", logging_file_string)
    if matches:
        ans = float(matches[-1])
    else:
        ans = 0
        
    return {"number_of_single_cut_vias": ans}
    
def max_overcon(logging_file_string):
    matches = re.findall(r"Max overcon =\s*([\d.]+)", logging_file_string)
    if matches:
        ans = float(matches[-1])
    else:
        ans = 0
        
    return {"max_overcon": ans}
    
def total_overcon(logging_file_string):
    matches = re.findall(r"Total overcon =\s*([\d.]+)", logging_file_string)
    if matches:
        ans = float(matches[-1])
    else:
        ans = 0
        
    return {"total_overcon": ans}
    
def worst_layer_gcell_overcon_rate(logging_file_string):
    matches = re.findall(r"Worst layer Gcell overcon rate =\s*([\d.]+)", logging_file_string)
    if matches:
        ans = float(matches[-1])
    else:
        ans = 0
        
    return {"worst_layer_gcell_overcon_rate": ans}

def hard_to_access_pins_ratio(logging_file_string: str) -> dict:
    # Define regex to extract the number of difficult-to-access instances
    difficult_access_pattern = re.search(r'(\d+) out of (\d+)\(\d+\.\d+%\) instances may be difficult to be accessed', logging_file_string)

    if difficult_access_pattern:
        difficult_instance_count = int(difficult_access_pattern.group(1))
        total_instance_count = int(difficult_access_pattern.group(2))
        
        # Calculate the ratio
        hard_to_access_ratio = difficult_instance_count / total_instance_count
    else:
        # If the pattern is not found, return None or 0 as per your design choice
        hard_to_access_ratio = None

    return {"hard_to_access_pins_ratio": hard_to_access_ratio}

def instance_blockages_count(logging_file_string: str) -> dict:
    # Use regex to find the instance blockages count from the log
    match = re.search(r'#Instance Blockages\s*:\s*(\d+)', logging_file_string)
    
    # Extract the value if the pattern is found
    feature_value = int(match.group(1)) if match else None

    return {"instance_blockages_count": feature_value}

def early_gr_overflow_percentage(logging_file_string: str) -> dict:
    # Define a regex pattern to capture both horizontal and vertical early global routing overflows
    pattern = r"Early Global Route overflow of layer group 1:\s*([\d.]+)% H \+ ([\d.]+)% V"
    
    # Search the string using the regex pattern
    match = re.search(pattern, logging_file_string)
    
    if match:
        # Extract horizontal and vertical overflow percentages as floats
        horizontal_overflow = float(match.group(1))
        vertical_overflow = float(match.group(2))
        
        # Calculate the total overflow as the sum of horizontal and vertical overflows
        total_overflow = horizontal_overflow + vertical_overflow
        
        # Return the total overflow percentage
        return {"early_gr_overflow_percentage": total_overflow}
    else:
        # Return None or some default value if no match is found
        return {"early_gr_overflow_percentage": 0}
    
def horizontal_overflow_percentage(logging_file_string: str) -> dict:
    # Define a regex pattern to capture the horizontal overflow percentage
    pattern = r"Overflow after Early Global Route\s*(\d*\.\d*)%\s*H"

    # Search the log string for the overflow line
    match = re.search(pattern, logging_file_string)

    # Extract the horizontal overflow percentage if a match is found
    if match:
        horizontal_overflow = float(match.group(1))
    else:
        horizontal_overflow = 0.0  # or handle as needed if no match is found

    return {"horizontal_overflow_percentage": horizontal_overflow}

def congestion_prediction_accuracy(logging_file_string: str) -> dict:

    # Regular expression to find reported congestion in the Early Global Routing section
    early_gr_congestion_match = re.search(r'Early Global Route overflow of layer group 1: (\d+\.\d+)% H \+ (\d+\.\d+)% V', logging_file_string)
    
    # Regular expression to find actual congestion after Global Routing
    global_routing_congestion_match = re.search(r'Overflow after GR: (\d+\.\d+)% H \+ (\d+\.\d+)% V', logging_file_string)
    
    if early_gr_congestion_match and global_routing_congestion_match:
        # Extract early congestion values
        early_h = float(early_gr_congestion_match.group(1))
        early_v = float(early_gr_congestion_match.group(2))
        
        # Extract actual congestion values
        actual_h = float(global_routing_congestion_match.group(1))
        actual_v = float(global_routing_congestion_match.group(2))
        
        # Calculate the accuracy as the inverse of the absolute difference normalized by the predicted values
        accuracy_h = 1 - abs(early_h - actual_h) / (early_h if early_h != 0 else 1)
        accuracy_v = 1 - abs(early_v - actual_v) / (early_v if early_v != 0 else 1)
        
        # Average out horizontal and vertical accuracies for a single congestion prediction accuracy value
        congestion_prediction_accuracy_value = (accuracy_h + accuracy_v) / 2
    else:
        # If there is an issue in finding the matches, we return a default accuracy of 0 or None
        congestion_prediction_accuracy_value = 0
    
    return {"congestion_prediction_accuracy": congestion_prediction_accuracy_value}

def initial_placement_efficiency(logging_file_string: str) -> dict:
    # Extracting Early Global Route overflow rate, congestion by layer (eGR), and total overcon congestion metrics
    early_gr_overflow_match = re.search(r'Early Global Route overflow.*?(\d+\.\d+)% H.*?(\d+\.\d+)% V', logging_file_string)
    global_route_max_overcon_match = re.search(r'Max overcon.*?=\s(\d+)\s*tracks\.', logging_file_string)
    global_route_total_overcon_match = re.search(r'Total overcon.*?=\s(\d+\.\d+)%\.', logging_file_string)

    # Default values if matches are not found
    early_gr_horizontal = 0.0
    early_gr_vertical = 0.0
    max_overcon_tracks = 0
    total_overcon_percent = 0.0

    if early_gr_overflow_match:
        early_gr_horizontal = float(early_gr_overflow_match.group(1))
        early_gr_vertical = float(early_gr_overflow_match.group(2))

    if global_route_max_overcon_match:
        max_overcon_tracks = int(global_route_max_overcon_match.group(1))
    
    if global_route_total_overcon_match:
        total_overcon_percent = float(global_route_total_overcon_match.group(1))

    # Calculating the initial placement efficiency as an inverse function of congestion metrics
    # The base metric considers both early and overall congestion factors.
    feature_value = 1 / (1 + early_gr_horizontal + early_gr_vertical + total_overcon_percent + max_overcon_tracks)

    return {"initial_placement_efficiency": feature_value}

def area_based_congestion_density(logging_file_string: str) -> dict:
    # Extract congestion data for layers
    layer_congestion_matches = re.findall(r'\s*M(\d+)\s+\((\d+-?\d*)\)\s+(\d+)\((.*?)%\)\s+(\d+)?\((.*?)%\)?\s+(\d+)?\((.*?)%\)?', logging_file_string)
    
    total_congestion_percentages = []
    
    for match in layer_congestion_matches:
        _, _, _, perc_1, _, perc_2, _, perc_3 = match
        # Convert percentages to float and add them
        percentage_values = [float(p.replace('%', '')) for p in [perc_1, perc_2, perc_3] if p.strip()]
        total_congestion_percent = sum(percentage_values)
        total_congestion_percentages.append(total_congestion_percent)
    
    # Calculate the overall area-based congestion density
    if total_congestion_percentages:
        area_based_congestion_density_value = sum(total_congestion_percentages) / len(total_congestion_percentages)
    else:
        area_based_congestion_density_value = 0.0
    
    return {"area_based_congestion_density": area_based_congestion_density_value}


def multi_layer_pin_access_variability(logging_file_string: str) -> dict:
    # Use regex to capture the percentage of Gcell for congestion per layer
    congestion_pattern = re.compile(r'M(\d+).*\((\d+\.\d+)%\)')
    layer_congestion = congestion_pattern.findall(logging_file_string)

    # Extract layer numbers and their respective congestion percentages
    layer_congestion_dict = {int(layer): float(congestion) for layer, congestion in layer_congestion}

    # Calculate the standard deviation of congestion percentages as a measure of variability
    if len(layer_congestion_dict) > 1:
        mean_congestion = sum(layer_congestion_dict.values()) / len(layer_congestion_dict)
        variance = sum((value - mean_congestion) ** 2 for value in layer_congestion_dict.values()) / len(layer_congestion_dict)
        variability = variance ** 0.5  # Standard deviation
    else:
        variability = 0

    # The feature is the calculated variability
    return {"multi_layer_pin_access_variability": variability}


def average_layer_congestion(logging_file_string: str) -> dict:
    # Regex pattern to extract congestion percentage for each layer
    congestion_pattern = re.compile(r'M(\d+)\s+\(.+?\)\s+\d+\(\s*([\d.]+)%\)')

    # Find all matches
    matches = congestion_pattern.findall(logging_file_string)

    # Calculate the average congestion
    if matches:
        congestion_values = [float(match[1]) for match in matches]
        average_congestion = sum(congestion_values) / len(congestion_values)
    else:
        average_congestion = 0.0

    return {"average_layer_congestion": average_congestion}

def pin_density_variance_map(logging_file_string: str) -> dict:
    # Find overcongested Gcells by layer
    congestion_info = re.findall(r'#  M\d\s+([^\n]+%)', logging_file_string)

    # Extract percentages for overcongested Gcells
    overcon_list = []
    for info in congestion_info:
        match = re.findall(r'\(([\d.]+)%\)', info)
        if match:
            overcon_percentages = list(map(float, match))
            overcon_list.extend(overcon_percentages)
    
    # Compute variance of these over-congestion percentages as a proxy for pin density variance
    if overcon_list:
        variance = np.var(overcon_list)
    else:
        variance = 0.0
    
    return {"pin_density_variance_map": variance}

def non_default_routing_rule_usage(logging_file_string: str) -> dict:
    # Use regex to find lines related to non-default routing rules (NDR)
    ndr_pattern = r"Total number of nets with non-default rule or having extra spacing = (\d+)"
    matches = re.search(ndr_pattern, logging_file_string)

    # Initialize feature value
    feature_value = 0

    # If a match is found, convert it from string to integer and assign to feature_value
    if matches:
        feature_value = int(matches.group(1))

    return {"non_default_routing_rule_usage": feature_value}

def crosstalk_sensitive_zones(logging_file_string: str) -> dict:
    import re
    
    # Initialize the counter for potential crosstalk sensitive zones
    potential_crosstalk_sensitive_zones = 0

    # Define regex pattern to capture warnings about pin access issues due to obstructions, which may imply crosstalk problems
    pattern = r'Pin access impeded near Instance .+? and Instance .+?\. Please inspect the area near the pin for any obstacle\.'

    # Search log for all instances of the pattern that imply indirect congestion due to access issues
    matches = re.findall(pattern, logging_file_string)

    # Count occurrences, each could represent a potentially crosstalk sensitive zone
    potential_crosstalk_sensitive_zones = len(matches)

    # Return as the feature dictionary
    return {"crosstalk_sensitive_zones": potential_crosstalk_sensitive_zones}


def inter_macro_channel_congestion(logging_file_string: str) -> dict:
    """
    Extracts the inter-macro channel congestion feature from the routability log.
    
    Args:
    - logging_file_string: A string containing the full text of the routability log.
    
    Returns:
    - A dictionary containing the feature 'inter_macro_channel_congestion' with its scalar value.
    """
    
    # Regex to find the congestion section and extract congestion percentages for each layer
    congestion_regex = r"#  M[2-4]\s*.*?([0-9]+\.[0-9]+)%"
    congestion_values = [float(match) for match in re.findall(congestion_regex, logging_file_string)]

    # Calculate the feature as the maximum of these congestion values
    if congestion_values:
        feature_value = max(congestion_values)
    else:
        feature_value = 0.0
    
    return {"inter_macro_channel_congestion": feature_value}


feat_func_list = [total_wirelength,
 number_vias,
 number_of_multi_cut_vias,
 number_of_single_cut_vias,
 max_overcon,
 total_overcon,
 worst_layer_gcell_overcon_rate,
 hard_to_access_pins_ratio,
 instance_blockages_count,
 early_gr_overflow_percentage,
 horizontal_overflow_percentage,
 congestion_prediction_accuracy,
 initial_placement_efficiency,
 area_based_congestion_density,
 multi_layer_pin_access_variability,
 average_layer_congestion,
 pin_density_variance_map,
 non_default_routing_rule_usage,
 crosstalk_sensitive_zones,
 inter_macro_channel_congestion]