import requests, base64
import json
import os

from PIL import Image
from io import BytesIO

def make_gpt4o_requests(image_path, prompts, max_tokens, temperature, top_p, data=None): 
    base64_images = []
    invoke_url = "https://api.openai.com/v1/chat/completions"
    api_key = "sk-proj-BvV8KNWJ6BHvDUaAqnoHT3BlbkFJ5AYbPXHZfPrtkmKfVRdL"
    if data is not None:
        for image in data['feature']:
            image = Image.fromarray(image.squeeze()*255).convert('RGB')
            buff = BytesIO()
            image.save(buff, format="PNG")
            buff.seek(0)
            image_b64 = base64.b64encode(buff.read()).decode()
            base64_images.append(image_b64)
    else:
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()
            base64_images.append(image_b64)
    # print(base64_images)
    headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
            }

    payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_images[0]}"
                                }
                            },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_images[1]}"
                                }
                            },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_images[2]}"
                                }
                            },
                        {
                            "type": "text",
                            "text": prompts,
                            },

                        ]
                    }
                ],
            "max_tokens": 50,
            }
    response = requests.post(invoke_url, headers=headers, json=payload)
    content = ""
    answer = ""
    try:
        answer = response.json()['choices'][0]['message']['content']
        answer = answer.split('<answer>')[1].split('</answer>')[0]
    except: 
        answer = 'No'
    return answer
