from dataset.load_dataset import load_dataset
from llm_inference.llama_requests import make_llama_requests
import argparse
from PIL import Image

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
    dataset = load_dataset(
            args.task, 
            args.ann_file, 
            args.data_root, 
            return_dict=True
    )
    for data in dataset:
        results = make_llama_requests(
            image_path=None,
            prompts=args.content,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            image=data["label"]  
        )
        print(results)

## python routability_IR_drop.py --max_token 512 --temperature 0.6 --top_p 0.9 --content "Describe this picture"
