import torch
import tqdm
import copy
import random
import numpy as np
from models.adapters import Adapter
from models.unet import UNet
from transformers import get_scheduler
from transformers import AutoTokenizer, PretrainedConfig
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from PIL import Image
from torchvision.transforms.functional import rgb_to_grayscale

# import clip
def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


class diffusion_inference:
    def __init__(self, model_id, unet_path, adpater_path, device):
        self.device = device
        self.model_id = model_id

        # load unet model
        self.scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
        self.model = UNet.from_pretrained(model_id, subfolder="unet").to(self.device)
        self.model.load_state_dict(torch.load(unet_path, weights_only=True, map_location=self.device))
        self.model.eval()
        
        self.scheduler.set_timesteps(100)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, subfolder="tokenizer", revision=None, use_fast=False
        )

        # import correct text encoder class
        text_encoder_cls = import_model_class_from_model_name_or_path(
            self.model_id, None
        )
        # Load scheduler and models
        self.text_encoder = text_encoder_cls.from_pretrained(
            self.model_id, subfolder="text_encoder", revision=None
        ).to(self.device)


        self.vae = AutoencoderKL.from_pretrained(
                self.model_id,
                subfolder="vae",
                revision=None,
            ).to(self.device)
        
        self.adapter = Adapter(cin=3*64, channels=[320, 640, 1280, 1280][:4], nums_rb=2, ksize=1, sk=True, use_conv=False)
        self.adapter.load_state_dict(torch.load(adpater_path, weights_only=True, map_location=self.device))
        self.adapter.to(self.device)
        self.adapter.eval()

    def reset_schedule(self, timesteps):
        self.scheduler.set_timesteps(timesteps)
        
    def get_adapter_features(self, image):
        with torch.no_grad():
            features = self.adapter(image)
        return features
        
    def inference(self, prompt, adapter_features=None, size=(256,256), guidance_scale=7.5, seed=-1, steps=100):
        prompt_embeds = self.compute_embeddings(
            prompt_batch=prompt, text_encoder=self.text_encoder, tokenizer=self.tokenizer
        )
        bsz = prompt_embeds["prompt_embeds"].shape[0]
        self.reset_schedule(steps)
        noisy_latents = torch.randn((bsz, 4, size[0]//8, size[1]//8)).to(self.device)

        with torch.no_grad():
            for t in tqdm.tqdm(self.scheduler.timesteps):
                with torch.no_grad():
                    input = noisy_latents
                    noise_pred = self.model(
                            input,
                            t,
                            encoder_hidden_states=prompt_embeds["prompt_embeds"],
                            down_block_additional_residuals=copy.deepcopy(adapter_features),
                    )[0]
                    noisy_latents = self.scheduler.step(noise_pred, t, noisy_latents)[0]

        image = self.vae.decode(noisy_latents / self.vae.config.scaling_factor, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        images = rgb_to_grayscale(image.detach().cpu())

        return images

    def encode_prompt(self, prompt_batch, text_encoder, tokenizer, device):
        captions = []
        for caption in prompt_batch:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, list):
                captions.extend(caption)
            else:
                raise ValueError(f"Unsupported caption type: {type(caption)}")
            
        with torch.no_grad():
            text_inputs = tokenizer(
                captions,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            
            if hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None
            
            text_input_ids = text_inputs.input_ids.to(device)
            prompt_embeds = text_encoder(
                text_input_ids,
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            
        return prompt_embeds


    def compute_embeddings(self, prompt_batch, text_encoder, tokenizer):
        prompt_embeds = self.encode_prompt(prompt_batch, text_encoder, tokenizer, device=self.device)
        prompt_embeds = prompt_embeds.to(self.device)
        
        return {"prompt_embeds": prompt_embeds}