def create_prompt(with_grid=True, with_distance=True):
    """
    Dynamically generates prompts for IR Drop Prediction tasks.

    Parameters:
    - task_type (str): Type of task ('all_features', 'without_densities', 'without_grids').
    - include_density (bool): Whether to include density information in the prompt.
    - include_grids (bool): Whether to include grid information in the prompt.

    Returns:
    - str: A formatted prompt for the specified task.
    """
    # Task description
    task_description = """
    ### Task
    These images are used for IR Drop Prediction in electronic design automation (EDA). 

    ### Task Description
    IR drop is defined as the deviation of voltage from the reference (VDD, VSS), and it has to be restricted to avoid degradation in timing and functionality. 
    IR drop values on each node from a vectorless power rail analysis are merged into corresponding tiles to form IR drop maps.
    The IR drop map provides the maximum IR drop value in each tile.
    """
    
    input_description = """
    ### Input
    You are given a pair of IR Drop Maps. Each map represents the severity of IR Drop on a chip, where high-severity areas are visually highlighted. 
    The task is to analyze and compare these maps based on the given features to determine which map exhibits a more severe IR Drop.
    """

    # Add grid information if applicable
    if with_grid:
        input_description += """
        The grid divides the image into smaller regions, each represented by a bounding box. 
        These bounding boxes are color-coded to indicate the severity of IR Drop:
        - **Red**: High severity
        - **Green**: Medium severity
        - **Blue**: Low severity
        
        By analyzing the distribution and density of these highlighted regions, the grid helps localize areas of concern, 
        quantify severity levels, and identify spatial patterns of IR Drop across the chip.
        """
    # Add distance metrics info 
    if with_distance:
        input_description += """
        For the 'high' severity level, distance metrics are provided to analyze the spatial relationships
        between clusters and the extent of the affected area:
        - **Mean Distance:** The average distance between all clusters of the high severity level. 
        A smaller mean distance indicates higher density, which represents a more severe condition.
        - **Total Red Area:** The total pixel area classified as high severity. A larger total red area indicates a more severe condition.
        """

    # Instruction part
    instruction = """
    ### Instruction
    Your task is to act as an EDA engineer to classify which IR Drop map has more severe IR Drop between the first or the second image.
    Return either 1 or 2 to indicate which IR Drop map is more severe without any other comments.
    """

    # Combine sections into the final prompt
    prompt = task_description + input_description + instruction
    return prompt


