CROSS_OVER_PROMPT = """In the first 7 images I give to you, the image features are (1)internal power(power_i), (2)switching power(power_s), (3)toggle rate scaled power(power_sca), (4)power_all = power_i + power_s + power_sca, (5)the power at time step 6(power_t_6), (6)the power at time step 13(power_t_13), (7)the power at time step 19(power_t_19).
These first 7 images can help engineers to understand the IR Drop and the routing demand of the layout.
The last image is the IR Drop map of the layout. 
These image data are belong to the digital designs, and they are based on RISC-V designs and 28nm planar technology.
The dataset are generated from 6 RTL designs with variations with variations in synthesis and physical design as show in below:

- RISCY-a:   Macros: 3/4/5, Frequency: 50/200/500MHz, Utilizations: 70/75/80/85/90%, Macro placement: 3, Power mesh setting: 8
- RISCY-FPU-a:  Macros: 3/4/5,  Frequency: 50/200/500MHz, Utilizations: 70/75/80/85/90%, Macro placement: 3, Power mesh setting: 8
- zero-riscy-a:  Macros: 3/4/5,  Frequency: 50/200/500MHz, Utilizations: 70/75/80/85/90%, Macro placement: 3, Power mesh setting: 8
- RISCY-b:  Macros: 13/14/15,  Frequency: 50/200/500MHz, Utilizations: 70/75/80/85/90%, Macro placement: 3, Power mesh setting: 8
- RISCY-FPU-b:  Macros: 13/14/15,  Frequency: 50/200/500MHz, Utilizations: 70/75/80/85/90%, Macro placement: 3, Power mesh setting: 8
- zero-riscy-b:  Macros: 13/14/15,  Frequency: 50/200/500MHz, Utilizations: 70/75/80/85/90%, Macro placement: 3, Power mesh setting: 8

Could you please identify and describe the features surrounding that these first 7 images as much as possible (location features, relation features, geometry features, proximity features)
I will provide you with an example feature list format and their definitions, please give me more digital features and their definitions. Remove duplicate features if they can be merged by the value. and request your response in JSON format.
The new features need to be helpful for generating the IR Drop map (last image). 
Don't generate the new features that are similar to the existing features.

Existing features:
{existing_features}

Example feature list with their definitions:
{feat_pool}

Now, only generated 10 new features and their definitions in the json format."""


MUTATION_PROMPT = """In the first 7 images I give to you, the image features are (1)internal power(power_i), (2)switching power(power_s), (3)toggle rate scaled power(power_sca), (4)power_all = power_i + power_s + power_sca, (5)the power at time step 6(power_t_6), (6)the power at time step 13(power_t_13), (7)the power at time step 19(power_t_19).
These first 7 images can help engineers to understand the IR Drop and the routing demand of the layout.
The last image is the IR Drop map of the layout. 
These image data are belong to the digital designs, and they are based on RISC-V designs and 28nm planar technology.
The dataset are generated from 6 RTL designs with variations with variations in synthesis and physical design as show in below:

- RISCY-a:   Macros: 3/4/5, Frequency: 50/200/500MHz, Utilizations: 70/75/80/85/90%, Macro placement: 3, Power mesh setting: 8
- RISCY-FPU-a:  Macros: 3/4/5,  Frequency: 50/200/500MHz, Utilizations: 70/75/80/85/90%, Macro placement: 3, Power mesh setting: 8
- zero-riscy-a:  Macros: 3/4/5,  Frequency: 50/200/500MHz, Utilizations: 70/75/80/85/90%, Macro placement: 3, Power mesh setting: 8
- RISCY-b:  Macros: 13/14/15,  Frequency: 50/200/500MHz, Utilizations: 70/75/80/85/90%, Macro placement: 3, Power mesh setting: 8
- RISCY-FPU-b:  Macros: 13/14/15,  Frequency: 50/200/500MHz, Utilizations: 70/75/80/85/90%, Macro placement: 3, Power mesh setting: 8
- zero-riscy-b:  Macros: 13/14/15,  Frequency: 50/200/500MHz, Utilizations: 70/75/80/85/90%, Macro placement: 3, Power mesh setting: 8

Please slightly mutate the feature into a new related digital feature surrounding that these first 7 images as much as possible (can be location features, relation features, geometry features, proximity features, but not limited to).
Only one new related feature is allowed. Don't generate the new feature with definition that is in other words. and request your response in JSON format.
The new feature need to be helpful for generating the IR Drop map (last image). 
Don't generate the new features that are similar to the existing features.

Existing features:
{existing_features}

Original feature:
{feature}
New mutated feature:
"""

DEDUPLICATION_PROMPT = """In the first 7 images I give to you, the image features are (1)internal power(power_i), (2)switching power(power_s), (3)toggle rate scaled power(power_sca), (4)power_all = power_i + power_s + power_sca, (5)the power at time step 6(power_t_6), (6)the power at time step 13(power_t_13), (7)the power at time step 19(power_t_19).
These first 7 images can help engineers to understand the IR Drop and the routing demand of the layout.
The last image is the IR Drop map of the layout. 
Act as a feature engineering expert, please help to deduplicate the features in the new feature list compared to the original feature list.
If a new feature sharing a similar definition or calculation method with the original one, then deduplicate the new one.

Original feature list and their definitions:
{feat_pool}

New feature list and their definitions:
{new_feat_pool}

Now, only provide the similar features in the new feature list in the json format:
Similar means both features are calculated on the same or similar logic and on the same images.
If the features are in other words on definitions, it is considered as duplicated.

[{{
  "feature": "feature_name",
  "reason": "reason for deduplication" 
}}]
"""

CODE_GEN_PROMPT = """In the first 7 images I give to you, the image features are (1)internal power(power_i), (2)switching power(power_s), (3)toggle rate scaled power(power_sca), (4)power_all = power_i + power_s + power_sca, (5)the power at time step 6(power_t_6), (6)the power at time step 13(power_t_13), (7)the power at time step 19(power_t_19).
These first 7 images can help engineers to understand the IR Drop and the routing demand of the layout.
The last image is the IR Drop map of the layout. 
These image data are belong to the digital designs, and they are based on RISC-V designs and 28nm planar technology.
The dataset are generated from 6 RTL designs with variations with variations in synthesis and physical design as show in below:

- RISCY-a:   Macros: 3/4/5, Frequency: 50/200/500MHz, Utilizations: 70/75/80/85/90%, Macro placement: 3, Power mesh setting: 8
- RISCY-FPU-a:  Macros: 3/4/5,  Frequency: 50/200/500MHz, Utilizations: 70/75/80/85/90%, Macro placement: 3, Power mesh setting: 8
- zero-riscy-a:  Macros: 3/4/5,  Frequency: 50/200/500MHz, Utilizations: 70/75/80/85/90%, Macro placement: 3, Power mesh setting: 8
- RISCY-b:  Macros: 13/14/15,  Frequency: 50/200/500MHz, Utilizations: 70/75/80/85/90%, Macro placement: 3, Power mesh setting: 8
- RISCY-FPU-b:  Macros: 13/14/15,  Frequency: 50/200/500MHz, Utilizations: 70/75/80/85/90%, Macro placement: 3, Power mesh setting: 8
- zero-riscy-b:  Macros: 13/14/15,  Frequency: 50/200/500MHz, Utilizations: 70/75/80/85/90%, Macro placement: 3, Power mesh setting: 8

Act as a feature engineering expert, please help to generate the code snippet using OpenCV for extracting the feature from these seven images.
The seven images are in grayscale, all images arrays are in [0-1].

Feature and its definition:
{feature}

Complete the following function code:

```python
def {feature_name}(images):
    power_i = images[0]
    power_s = images[1]
    power_sca = images[2]
    power_all = images[3]
    power_t_6 = images[4]
    power_t_13 = images[5]
    power_t_19 = images[6]

    
    # Todo: Your code start here
    
    
    return {{"feature_name": feature_value}}
```
"""