import argparse
import requests
import numpy as np
import os

from save_samples import save_sample_images
from load_images import load_images
from distance_metrics import compute_distance_metrics_multilevel
from create_prompt import create_prompt
from create_payload import create_payload_pair
from display import display_array

# put openai api key here
api_key = "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"

def setup(num_pairs, feature_path, label_path):
    """
    Sets up the dataset for inference by loading images, masks, and computed metrics.
    """
    num_samples = num_pairs * 2

    # create pairwise folder
    pairwise_path = "./pairwise_samples"  
    os.makedirs(pairwise_path, exist_ok=True)  

    sample_arrs = save_sample_images(feature_path, label_path, pairwise_path, num_samples) 
    pair_images = load_images(pairwise_path)

    # load masks
    feature_masks = [sample_arr['feature_masks'] for sample_arr in sample_arrs]
    label_masks = [sample_arr['label_masks'] for sample_arr in sample_arrs]

    # load dicts 
    feature_metrics_dicts = [
        [compute_distance_metrics_multilevel(feature_mask)[0] for feature_mask in feature_mask_list]
        for feature_mask_list in feature_masks
    ]

    feature_centroids_dicts = [
        [compute_distance_metrics_multilevel(feature_mask)[1] for feature_mask in feature_mask_list]
        for feature_mask_list in feature_masks
    ]

    label_metrics_dicts = [compute_distance_metrics_multilevel(label_mask)[0] for label_mask in label_masks]
    label_centroids_dicts = [compute_distance_metrics_multilevel(label_mask)[1] for label_mask in label_masks]  # for display

    return pair_images, feature_metrics_dicts, label_metrics_dicts, label_centroids_dicts, sample_arrs


def inference(num_pairs, with_grid, with_distance, pair_images, feature_metrics_dicts, label_metrics_dicts, sample_arrs):
    """
    Runs inference on the given pairs of images using GPT-4o and evaluates performance.
    """
    prompt = create_prompt(with_grid=with_grid, with_distance=with_distance)
    
    # Choose label type
    min_distance_as_label = False
    mean_distance_as_label = False
    severity_score_as_label = True

    correct = 0
    result = {}
    bad_cases = []
    for i in range(0, num_pairs):
        start = i * 2
        end = start + 2
        pair_image = pair_images[start:end]
        pair_feature_metrics_dict = feature_metrics_dicts[start:end]
        pair_label_metrics_dict = label_metrics_dicts[start:end]

        # Determine target based on label type
        if min_distance_as_label:
            labels = [pair_label_metrics_dict[0]['high']['min_distance'], pair_label_metrics_dict[1]['high']['min_distance']]
            label_type = "min distance: "
            target = np.argmin(labels) + 1
        elif mean_distance_as_label:
            labels = [pair_label_metrics_dict[0]['high']['mean_distance'], pair_label_metrics_dict[1]['high']['mean_distance']]
            label_type = "mean distance: "
            target = np.argmin(labels) + 1
        elif severity_score_as_label:
            labels = [pair_label_metrics_dict[0]['high']['severity_score'], pair_label_metrics_dict[1]['high']['severity_score']]
            label_type = "severity score: "
            target = np.argmax(labels) + 1

        target_int = int(target)

        # Extract images for display
        label_arr_1 = sample_arrs[start:end][0]['label_arr'] 
        label_arr_2 = sample_arrs[start:end][1]['label_arr']
        label_arr_grid_1 = sample_arrs[start:end][0]['label_arr_grid'] 
        label_arr_grid_2 = sample_arrs[start:end][1]['label_arr_grid']
        
        # Prediction using GPT-4o
        payload = create_payload_pair(pair_image, prompt, pair_feature_metrics_dict, pair_label_metrics_dict, with_grid=with_grid, with_distance=with_distance)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        pred = response.json()['choices'][0]['message']['content']
        
        try:
            pred_int = int(pred.strip())
        except ValueError:
            print(f"Invalid prediction format: {pred}")
            pred_int = -1

        print(f'sample_{start} and sample_{end-1}')
        print('GPT-4o: ', pred_int)
        print('target: ', target_int)
        
        # Compare predictions
        if pred_int == target_int:
            correct += 1
        else:
            if with_grid:
                display_array(label_arr_grid_1)
                print(label_type, labels[0])
                print()
                display_array(label_arr_grid_2)
                print(label_type, labels[1])
            else:
                display_array(label_arr_1)
                print(label_type, labels[0])
                print()
                display_array(label_arr_2)
                print(label_type, labels[1])

            bad_case = {"sample": f"{start} and {end-1}",
                        "gpt-4o": pred_int,
                        "target:": target_int,
                        "severity_score1": labels[0],
                        "severity_score2": labels[1],
                        }
            bad_cases.append(bad_case)
        
        print("".join(['-'] * 50))
    
    result['number of pairs'] = num_pairs
    result['correct'] = correct
    result['accuracy'] = round(correct / num_pairs, 4)
    
    # # Write result and bad cases to files
    # with open("result.txt", "w") as result_file:
    #     for key, value in result.items():
    #         result_file.write(f"{key}: {value}\n")

    # with open("bad_cases.txt", "w") as bad_cases_file:
    #     for case in bad_cases:
    #         bad_cases_file.write(f"{case}\n")

    print(result)
    return result, bad_cases


if __name__ == "__main__":
    # Command-line parser setup
    parser = argparse.ArgumentParser(description="Run IR Drop inference with GPT-4o.")
    parser.add_argument("--num_pairs", type=int, required=True, help="Number of pairs to process.")
    parser.add_argument("--feature_path", type=str, required=True, help="Path to the feature images.")
    parser.add_argument("--label_path", type=str, required=True, help="Path to the label images.")
    parser.add_argument("--with_grid", action="store_true", help="Include grid information in the prompt.")
    parser.add_argument("--with_distance", action="store_true", help="Include distance metrics in the prompt.")

    args = parser.parse_args()

    # Setup and inference
    pair_images, feature_metrics_dicts, label_metrics_dicts, label_centroids_dicts, sample_arrs = setup(
        args.num_pairs, args.feature_path, args.label_path
    )
    inference(
        args.num_pairs, args.with_grid, args.with_distance,
        pair_images, feature_metrics_dicts, label_metrics_dicts, sample_arrs
    )

"""
example usage: 
python inference.py --num_pairs 3 --feature_path "/data2/NVIDIA/CircuitNet-N28/Dataset/IR_drop/feature" --label_path "/data2/NVIDIA/CircuitNet-N28/Dataset/IR_drop/label" --with_grid

"""