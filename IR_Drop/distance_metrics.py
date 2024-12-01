from math import sqrt
import numpy as np
from scipy.ndimage import label, center_of_mass
from scipy.spatial.distance import pdist, squareform

def compute_distance_metrics_multilevel(masks, normalize=True):
    """
    Computes severity score focusing on mean distance and total red area, excluding variance.
    
    Parameters:
    - masks: Dictionary containing binary masks for different severity levels.
    - normalize: Boolean to indicate whether to normalize scales of the parameters.
    
    Returns:
    - metrics_dict: Dictionary containing the computed metrics and severity score.
    """
    

    metrics_dict = {}
    centroids_dict = {}

    # Ensure masks is not a list (just handle the first mask)
    if isinstance(masks, list):
        masks = masks[0]

    # Process only the 'high' severity level
    if "high" in masks:
        mask = masks["high"]

        # Label clusters in the 'high' severity area
        labeled_array, num_clusters = label(mask)
        
        if num_clusters == 0:  # No clusters in the 'high' area
            metrics = {
                "mean_distance": float('nan'),
                "severity_score": 0.0,
                "total_red_area": 0,
            }
            centroids = None
        else:
            # Compute centroids
            centroids = np.array(center_of_mass(mask, labeled_array, range(1, num_clusters + 1)))
            
            # Compute pairwise distances
            if len(centroids) > 1:
                distance_matrix = squareform(pdist(centroids))
                np.fill_diagonal(distance_matrix, np.inf)
                finite_distances = distance_matrix[np.isfinite(distance_matrix)]
                mean_distance = np.mean(finite_distances)
            else:
                mean_distance = float('nan')  # No mean distance for a single cluster

            # Calculate total red area
            total_red_area = np.sum(mask)

            # Normalize parameters if normalization is enabled
            if normalize:
                mean_distance_max = 150  # Example max value for mean_distance
                total_red_area_max = 20000  # Example max value for total_red_area

                mean_distance_norm = mean_distance / mean_distance_max if mean_distance_max > 0 else 1
                total_red_area_norm = total_red_area / total_red_area_max if total_red_area_max > 0 else 1
            else:
                mean_distance_norm = mean_distance
                total_red_area_norm = total_red_area

            # Invert the normalized score for mean distance
            mean_distance_score = 1 - mean_distance_norm if not np.isnan(mean_distance_norm) else 0

            # Assign weights to each feature
            mean_distance_weight = 0.3
            total_red_area_weight = 0.7

            # Compute severity score with weights
            severity_score = (
                mean_distance_score * mean_distance_weight +
                total_red_area_norm * total_red_area_weight
            )

            # Metrics dictionary for the 'high' level
            metrics = {
                "mean_distance": round(mean_distance, 4) if len(centroids) > 1 else float('nan'),
                "severity_score": round(severity_score, 4),
                "total_red_area": total_red_area,
            }
        
        # Save metrics and centroids for the 'high' level
        metrics_dict["high"] = metrics
        centroids_dict["high"] = centroids
    else:
        # No 'high' level found in the masks
        metrics_dict["high"] = {
            "mean_distance": float('nan'),
            "severity_score": 0.0,
            "total_red_area": 0,
        }
        centroids_dict["high"] = None

    return metrics_dict, centroids_dict
