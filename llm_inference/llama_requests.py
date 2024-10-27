# --coding: utf-8 a--
import requests, base64
import json
import argparse
from PIL import Image
from io import BytesIO

def make_llama_requests(image_path, prompts, max_tokens, temperature, top_p, image=None): 
    invoke_url = "https://ai.api.nvidia.com/v1/gr/meta/llama-3.2-90b-vision-instruct/chat/completions"
    stream = True
    if image is not None:
        image = Image.fromarray(image.squeeze()).convert('RGB')
        buff = BytesIO()
        image.save(buff, format="PNG")
        buff.seek(0)
        image_b64 = base64.b64encode(buff.read()).decode()
    else:
        with open(image_path, "rb") as f:
          image_b64 = base64.b64encode(f.read()).decode()

    headers = {
      "Authorization": "Bearer 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",
      "Accept": "text/event-stream" if stream else "application/json"
    }

    payload = {
      "model": 'meta/llama-3.2-90b-vision-instruct',
      "messages": [
        {
          "role": "user",
          "content": f'{prompts} <img src="data:image/png;base64,{image_b64}" />'
        }
      ],
      "max_tokens": max_tokens,
      "temperature": temperature,
      "top_p": top_p,
      "stream": stream
    }

    response = requests.post(invoke_url, headers=headers, json=payload)
    content = ""
    
    if stream:
        for line in response.iter_lines():
            if line:
                # print(line.decode("utf-8"))
                j = json.loads(line.decode("utf-8")[6::])
                # print(j)
                content += j["choices"][0]["delta"]["content"]
                if j["choices"][0]["finish_reason"] == "stop":
                    break

    return content

def parse_args():
    parser = argparse.ArgumentParser()
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
    results = make_llama_requests(
        image_path=args.image_path,
        prompts=args.content,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    # print(results)

### Example Usage
# python llama_requests.py --max_token 512 --temperature 0.6 --top_p 0.9 --content "Describe this picture" --image_path ./image.png
