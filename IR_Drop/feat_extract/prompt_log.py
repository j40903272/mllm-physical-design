CROSS_OVER_PROMPT = """The dataset belongs to the digital designs, and is based on RISC-V designs and 28nm planar technology.
The dataset is generated from 6 RTL designs with variations in synthesis and physical design as show in below:

- RISCY-a:   Macros: 3/4/5, Frequency: 50/200/500MHz, Utilizations: 70/75/80/85/90%, Macro placement: 3, Power mesh setting: 8
- RISCY-FPU-a:  Macros: 3/4/5,  Frequency: 50/200/500MHz, Utilizations: 70/75/80/85/90%, Macro placement: 3, Power mesh setting: 8
- zero-riscy-a:  Macros: 3/4/5,  Frequency: 50/200/500MHz, Utilizations: 70/75/80/85/90%, Macro placement: 3, Power mesh setting: 8
- RISCY-b:  Macros: 13/14/15,  Frequency: 50/200/500MHz, Utilizations: 70/75/80/85/90%, Macro placement: 3, Power mesh setting: 8
- RISCY-FPU-b:  Macros: 13/14/15,  Frequency: 50/200/500MHz, Utilizations: 70/75/80/85/90%, Macro placement: 3, Power mesh setting: 8
- zero-riscy-b:  Macros: 13/14/15,  Frequency: 50/200/500MHz, Utilizations: 70/75/80/85/90%, Macro placement: 3, Power mesh setting: 8

Here is a sample routability log:
{routability_log}


Could you please identify and describe the features surrounding the routability log as much as possible.
I will provide you with an example feature list format and their definitions, please give me more digital features and their definitions. Remove duplicate features if they can be merged by the value. and request your response in JSON format.
The new features need to be helpful for generating the congestion map (last image). 
Don't generate the new features that are similar to the existing features.

Existing features:
{existing_features}

Example feature list with their definitions:
{feat_pool}

Now, only generated 10 new features and their definitions in the json format."""


MUTATION_PROMPT = """The dataset belongs to the digital designs, and is based on RISC-V designs and 28nm planar technology.
The dataset is generated from 6 RTL designs with variations in synthesis and physical design as show in below:

- RISCY-a:   Macros: 3/4/5, Frequency: 50/200/500MHz, Utilizations: 70/75/80/85/90%, Macro placement: 3, Power mesh setting: 8
- RISCY-FPU-a:  Macros: 3/4/5,  Frequency: 50/200/500MHz, Utilizations: 70/75/80/85/90%, Macro placement: 3, Power mesh setting: 8
- zero-riscy-a:  Macros: 3/4/5,  Frequency: 50/200/500MHz, Utilizations: 70/75/80/85/90%, Macro placement: 3, Power mesh setting: 8
- RISCY-b:  Macros: 13/14/15,  Frequency: 50/200/500MHz, Utilizations: 70/75/80/85/90%, Macro placement: 3, Power mesh setting: 8
- RISCY-FPU-b:  Macros: 13/14/15,  Frequency: 50/200/500MHz, Utilizations: 70/75/80/85/90%, Macro placement: 3, Power mesh setting: 8
- zero-riscy-b:  Macros: 13/14/15,  Frequency: 50/200/500MHz, Utilizations: 70/75/80/85/90%, Macro placement: 3, Power mesh setting: 8

Here is a sample routability log:
{routability_log}


Please slightly mutate the feature into a new related digital feature surrounding the routability log as much as possible.
Only one new related feature is allowed. Don't generate the new feature with definition that is in other words. and request your response in JSON format.
The new feature need to be helpful for generating the congestion map (last image). 
Don't generate the new feature that is similar to the existing features.

Existing features:
{existing_features}

Original feature:
{feature}
New mutated feature:
"""

DEDUPLICATION_PROMPT = """Act as a feature engineering expert, please help to deduplicate the features in the new feature list compared to the original feature list.
If a new feature sharing a similar definition or calculation method with the original one, then deduplicate the new one.

Original feature list and their definitions:
{feat_pool}

New feature list and their definitions:
{new_feat_pool}

Now, only provide the similar features in the new feature list in the json format:
Similar means both features are calculated on the same or similar logic.
If the features are in other words on definitions, it is considered as duplicated.

[{{
  "feature": "feature_name",
  "reason": "reason for deduplication" 
}}]
"""

CODE_GEN_PROMPT = """The dataset belongs to the digital designs, and is based on RISC-V designs and 28nm planar technology.
The dataset is generated from 6 RTL designs with variations in synthesis and physical design as show in below:

- RISCY-a:   Macros: 3/4/5, Frequency: 50/200/500MHz, Utilizations: 70/75/80/85/90%, Macro placement: 3, Power mesh setting: 8
- RISCY-FPU-a:  Macros: 3/4/5,  Frequency: 50/200/500MHz, Utilizations: 70/75/80/85/90%, Macro placement: 3, Power mesh setting: 8
- zero-riscy-a:  Macros: 3/4/5,  Frequency: 50/200/500MHz, Utilizations: 70/75/80/85/90%, Macro placement: 3, Power mesh setting: 8
- RISCY-b:  Macros: 13/14/15,  Frequency: 50/200/500MHz, Utilizations: 70/75/80/85/90%, Macro placement: 3, Power mesh setting: 8
- RISCY-FPU-b:  Macros: 13/14/15,  Frequency: 50/200/500MHz, Utilizations: 70/75/80/85/90%, Macro placement: 3, Power mesh setting: 8
- zero-riscy-b:  Macros: 13/14/15,  Frequency: 50/200/500MHz, Utilizations: 70/75/80/85/90%, Macro placement: 3, Power mesh setting: 8

Here is a sample routability log:
{routability_log}


Act as a feature engineering expert, please help to generate the code snippet using regex for extracting the feature from the routability log.
The feature must be a scalar value.

Feature and its definition:
{feature}

Complete the following function code:

```python
def {feature_name}(logging_file_string: str) -> dict:

    # Todo: Your code start here
    
    return {{"feature_name": feature_value}}
```
"""