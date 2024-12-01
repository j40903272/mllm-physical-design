def create_payload_pair(pair_images, prompt, pair_feature_metrics_dict, pair_label_metrics_dict, with_grid, with_distance):

    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": []
            }
        ],
        "max_tokens": 700
    }

    # set number of feature images
    num_feature = 0

    # Build the content dynamically
    content = []

    # add the prompt
    content.append({
    "type": "text",
    "text": prompt
    })
    
    # 2 samples in a pair
    for i, sample in enumerate(pair_images):
        feature_images = sample['feature_base64s']
        label_image = sample['label_base64']
        feature_images_grid = sample['feature_grid_base64s']
        label_image_grid = sample['label_grid_base64']

        # choosing feature
        if with_grid:
            images = feature_images_grid
            label_image = label_image_grid
        else:
            images = feature_images
            label_image = label_image

        if with_distance:
            distance_metrics = pair_feature_metrics_dict[i]
        
        for j, image in enumerate(images[:num_feature]):

            # Add the image block
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image}"
                }
            })

            if with_distance:
                content.append({
                    "type": "text",
                    "text": f"The distance metrics are: {distance_metrics[j]['high']['mean_distance']}"
                })

        if isinstance(label_image, list):
            label_image = label_image[0]
        
        # Add the label image
        content.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{label_image}"
        }
        })

        if with_distance:
            content.append({
                "type": "text",
                "text": f"The distance metrics are: {pair_label_metrics_dict[i]['high']['mean_distance']}, {pair_label_metrics_dict[i]['high']['total_red_area']}"
            })


        content.append({
        "type": "text",
        "text": f"This is the label image for this sample"
        })


    # Assign the generated content to the payload
    payload["messages"][0]["content"] = content

    return payload 