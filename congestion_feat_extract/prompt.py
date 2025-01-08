CROSS_OVER_PROMPT = """In the first 3 images I give to you, the first image is the region on the layout covered by macro, the second image is RUDY (Rectangular Uniform wire DensitY) which works as a early routing demand estimation after placement, the third image is RUDY pin.
These first 3 images can help engineers to understand the congestion of the layout and the routing demand of the layout.
The last image is the congestion map of the layout. 
These image data are belong to the digital designs, and they are based on RISC-V designs and 28nm planar technology.
The dataset are generated from 6 RTL designs with variations with variations in synthesis and physical design as show in below:

- RISCY-a:   Macros: 3/4/5, Frequency: 50/200/500MHz, Utilizations: 70/75/80/85/90%, Macro placement: 3, Power mesh setting: 8
- RISCY-FPU-a:  Macros: 3/4/5,  Frequency: 50/200/500MHz, Utilizations: 70/75/80/85/90%, Macro placement: 3, Power mesh setting: 8
- zero-riscy-a:  Macros: 3/4/5,  Frequency: 50/200/500MHz, Utilizations: 70/75/80/85/90%, Macro placement: 3, Power mesh setting: 8
- RISCY-b:  Macros: 13/14/15,  Frequency: 50/200/500MHz, Utilizations: 70/75/80/85/90%, Macro placement: 3, Power mesh setting: 8
- RISCY-FPU-b:  Macros: 13/14/15,  Frequency: 50/200/500MHz, Utilizations: 70/75/80/85/90%, Macro placement: 3, Power mesh setting: 8
- zero-riscy-b:  Macros: 13/14/15,  Frequency: 50/200/500MHz, Utilizations: 70/75/80/85/90%, Macro placement: 3, Power mesh setting: 8

Could you please identify and describe the features surrounding that these first 3 images as much as possible (location features, relation features, geometry features, proximity features)
I will provide you with an example feature list format and their definitions, please give me more digital features and their definitions. Remove duplicate features if they can be merged by the value. and request your response in JSON format.
The new features need to be helpful for generating the congestion map (last image). 
Don't generate the new features that are similar to the existing features.

Existing features:
{existing_features}

Example feature list with their definitions:
{feat_pool}

Now, only generated 10 new features and their definitions in the json format."""


MUTATION_PROMPT = """In the first 3 images I give to you, the first image is the region on the layout covered by macro, the second image is RUDY (Rectangular Uniform wire DensitY) which works as a early routing demand estimation after placement, the third image is RUDY pin.
These first 3 images can help engineers to understand the congestion of the layout and the routing demand of the layout.
The last image is the congestion map of the layout. 
These image data are belong to the digital designs, and they are based on RISC-V designs and 28nm planar technology.
The dataset are generated from 6 RTL designs with variations with variations in synthesis and physical design as show in below:

- RISCY-a:   Macros: 3/4/5, Frequency: 50/200/500MHz, Utilizations: 70/75/80/85/90%, Macro placement: 3, Power mesh setting: 8
- RISCY-FPU-a:  Macros: 3/4/5,  Frequency: 50/200/500MHz, Utilizations: 70/75/80/85/90%, Macro placement: 3, Power mesh setting: 8
- zero-riscy-a:  Macros: 3/4/5,  Frequency: 50/200/500MHz, Utilizations: 70/75/80/85/90%, Macro placement: 3, Power mesh setting: 8
- RISCY-b:  Macros: 13/14/15,  Frequency: 50/200/500MHz, Utilizations: 70/75/80/85/90%, Macro placement: 3, Power mesh setting: 8
- RISCY-FPU-b:  Macros: 13/14/15,  Frequency: 50/200/500MHz, Utilizations: 70/75/80/85/90%, Macro placement: 3, Power mesh setting: 8
- zero-riscy-b:  Macros: 13/14/15,  Frequency: 50/200/500MHz, Utilizations: 70/75/80/85/90%, Macro placement: 3, Power mesh setting: 8

Please slightly mutate the feature into a new related digital feature surrounding that these first 3 image as much as possible (can be location features, relation features, geometry features, proximity features, but not limited to).
Only one new related feature is allowed. Don't generate the new feature with definition that is in other words. and request your response in JSON format.
The new feature need to be helpful for generating the congestion map (last image). 
Don't generate the new feature that is similar to the existing features.

Existing features:
{existing_features}

Original feature:
{feature}
New mutated feature:
{{
  "feature_name": "feature_definition"
}}
"""

DEDUPLICATION_PROMPT = """In the first 3 images I give to you, the first image is the region on the layout covered by macro, the second image is RUDY (Rectangular Uniform wire DensitY) which works as a early routing demand estimation after placement, the third image is RUDY pin.
These first 3 images can help engineers to understand the congestion of the layout and the routing demand of the layout. 
The last image is the congestion map of the layout. 
Act as a feature engineering expert, please help to deduplicate the features in the new feature list compared to the original feature list.
If a new feature sharing a similar definition or calculation method with the original one, then deduplicate the new one.

Original feature list and their definitions:
{feat_pool}

New feature list and their definitions:
{new_feat_pool}

Now, only provide the similar features in the new feature list in the json format:
Similar means both features are calculated on the same or similar logic and on the same images(Congestion, RUDY, or RUDY-PIN).
If the features are in other words on definitions, it is considered as duplicated.

[{{
  "feature": "feature_name",
  "reason": "reason for deduplication" 
}}]
"""

CODE_GEN_PROMPT = """In the first 3 images I give to you, the first image is the region on the layout covered by macro, the second image is RUDY (Rectangular Uniform wire DensitY) which works as a early routing demand estimation after placement, the third image is RUDY pin.
These first 3 images can help engineers to understand the congestion of the layout and the routing demand of the layout.
The last image is the congestion map of the layout. 
These image data are belong to the digital designs, and they are based on RISC-V designs and 28nm planar technology.
The dataset are generated from 6 RTL designs with variations with variations in synthesis and physical design as show in below:

- RISCY-a:   Macros: 3/4/5, Frequency: 50/200/500MHz, Utilizations: 70/75/80/85/90%, Macro placement: 3, Power mesh setting: 8
- RISCY-FPU-a:  Macros: 3/4/5,  Frequency: 50/200/500MHz, Utilizations: 70/75/80/85/90%, Macro placement: 3, Power mesh setting: 8
- zero-riscy-a:  Macros: 3/4/5,  Frequency: 50/200/500MHz, Utilizations: 70/75/80/85/90%, Macro placement: 3, Power mesh setting: 8
- RISCY-b:  Macros: 13/14/15,  Frequency: 50/200/500MHz, Utilizations: 70/75/80/85/90%, Macro placement: 3, Power mesh setting: 8
- RISCY-FPU-b:  Macros: 13/14/15,  Frequency: 50/200/500MHz, Utilizations: 70/75/80/85/90%, Macro placement: 3, Power mesh setting: 8
- zero-riscy-b:  Macros: 13/14/15,  Frequency: 50/200/500MHz, Utilizations: 70/75/80/85/90%, Macro placement: 3, Power mesh setting: 8

Act as a feature engineering expert, please help to generate the code snippet using OpenCV for extracting the feature from these three images.
If the feature result is a length or an area, note that the unit needs to be um, and the image is 256x256 pixels, each pixel is 2.25umx2.25um.
The three images are in grayscale, all images arrays are in [0-1]. The macro image needs to be converted to [0-255]. Other images stay in [0-1].

Feature and its definition:
{feature}

Complete the following function code:

```python
def {feature_name}(images):
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
    
    # Todo: Your code start here
    
    
    return {{"feature_name": feature_value}}
```
"""