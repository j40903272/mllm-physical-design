from dataset.load_dataset import load_dataset
from llm_inference.llama_requests import make_llama_requests
import argparse
from PIL import Image
predicted_task = "Congestion Prediction"
prompt = f"""
### Task
These images are used for {predicted_task} in electronic design automation (EDA). 

### Task Description
Congestion is defined as the overflow of routing demand over available routing resource in the routing stage of the back-end design. It is frequently adopted as the metric to evaluate routability, i.e., the prospective quality of routing based on the current design solution. The congestion prediction is necessary to guide the optimization in placement stage and reduce total turn-around time.

### Input
The first image is called Macro Region feature.
The second image is called Rectangular Uniform wire Density (RUDY).
The third image is called RUDY pin.


### Instruction
Act as a congestion prediction model (don't use random).
Predict a congestion score based on the given images.

The congestion level is defined as follows:
- Low: 0 - 0.1
- Moderate: 0.1 - 0.15
- High: > 0.15

Only give the congestion level answer in <answer>...</answer> tag.
"""
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", 
        default = 'congestion',
        type = str,
        help = "Task Name",
        choices = ['congestion', 'IR_drop', 'DRC']
    )
    parser.add_argument(
        "--ann_file", 
        default = "dataset/data/train_N28.csv",
        type = str,
    )
    parser.add_argument(
        "--data_root", 
        default = "/data2/NVIDIA/CircuitNet-N28/Dataset",
        type = str,
    )
    parser.add_argument(
        "--max_tokens", 
        default = 512,
        type = int,
        help = "The max_tokens parameter of Llama 3.2"
    )

    parser.add_argument(
        "--temperature", 
        default = 0.3,
        type = float,
        help = "The temprature parameter of Llama 3.2"
    )
    
    parser.add_argument(
        "--top_p", 
        default = 0.8,
        type = float,
        help = "The top_p parameter of Llama 3.2"
    )
    
    parser.add_argument(
        "--content", 
        default = "What is in this image?",
        type = str,
        help = "The prompt used to ask Llama 3.2"
    )
    
    parser.add_argument(
        "--image_path", 
        default = "./image.png",
        type = str,
        help = "The image used to ask Llama 3.2"
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(prompt)
    dataset = load_dataset(
            args.task, 
            args.ann_file, 
            args.data_root, 
            return_dict=True
    )
    for data in dataset:
        results = make_llama_requests(
            image_path=None,
            # prompts=args.content,
            prompts=prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            image=data["label"]  
        )
        print(results)

## python routability_IR_drop_inferencing.py --max_token 512 --temperature 0.6 --top_p 0.9 --content "Describe this picture"
