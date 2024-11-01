# DreamStyler
# Copyright (c) 2024-present NAVER Webtoon
# Apache-2.0

import os
from os.path import join as ospj
import click
import torch
import imageio
import numpy as np
from PIL import Image
from diffusers import ControlNetModel, UniPCMultistepScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from controlnet_aux.processor import Processor
import custom_pipelines
import argparse



def load_model(sd_path, controlnet_path, embedding_path, placeholder_token="<sks1>", num_stages=6):
    tokenizer = CLIPTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder")
    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)

    placeholder_token = [f"{placeholder_token}-T{t}" for t in range(num_stages)]
    num_added_tokens = tokenizer.add_tokens(placeholder_token)
    if num_added_tokens == 0:
        raise ValueError("The tokens are already in the tokenizer")
    placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)
    text_encoder.resize_token_embeddings(len(tokenizer))

    learned_embeds = torch.load(embedding_path)
    token_embeds = text_encoder.get_input_embeddings().weight.data
    for token, token_id in zip(placeholder_token, placeholder_token_id):
        token_embeds[token_id] = learned_embeds[token]

    pipeline = custom_pipelines.StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        sd_path,
        controlnet=controlnet,
        text_encoder=text_encoder.to(torch.float16),
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.enable_model_cpu_offload()
    processor_midas = Processor("depth_midas")

    return pipeline, processor_midas



def get_options():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sd_path", type=str, default='../../resources/models/sd-v1-4')
    parser.add_argument("--controlnet_path", type=str, default="lllyasviel/control_v11f1p_sd15_depth")
    parser.add_argument("--embedding_path", type=str, default='./trained_logs') 
    parser.add_argument("--save_path", type=str, default="../../stylized_images/Dreamstyler")
    parser.add_argument("--prompt", type=str, default="{}")
    parser.add_argument("--placeholder_token", type=str, default="<sks03>")
    parser.add_argument("--num_stages", type=int, default=6)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--save_name", type=str, default='save')
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--content_path", type=str, default='../../dataset/content/custom5')
    parser.add_argument("--style_path", type=str, default="../../dataset/style/wikiart")
    parser.add_argument("--style_nums", type=str, default="10") #52,20,38,83,79 - wikiart // 1003,1004,1005 - inst
    
    opt = parser.parse_args()
    
    return opt


def style_transfer(opt):
    generator = None if opt.seed is None else torch.Generator(device="cuda").manual_seed(opt.seed)
    cross_attention_kwargs = {"num_stages": opt.num_stages}
    
    
   
    content_name = opt.content_path.split('/')[-1]
    
    for style_num in sorted(opt.style_nums.split(',')):
        embedding_path = os.path.join(opt.embedding_path, f'style_{style_num}', '/embedding/final.bin'.lstrip("/"))
        save_path = os.path.join(opt.save_path, f'content_{content_name}', f'style_{style_num}')
        os.makedirs(save_path, exist_ok=True)
        
        pipeline, processor = load_model(
            opt.sd_path,
            opt.controlnet_path,
            embedding_path,
            opt.placeholder_token,
            opt.num_stages,
        )
        
        for file in sorted(os.listdir(opt.content_path)):
            content_image_name = file
            content_path = os.path.join(opt.content_path, file)
            content_image = Image.open(content_path)
            content_image = content_image.resize((opt.resolution, opt.resolution))
            control_image = processor(content_image, to_pil=True)
            pos_prompt = [opt.prompt.format(f"{opt.placeholder_token}-T{t}") for t in range(opt.num_stages)]

            outputs = []
            torch.manual_seed(1)    
            
            output = pipeline(
                prompt=pos_prompt,
                num_inference_steps=30, 
                generator=generator,
                image=content_image,
                control_image=control_image,
                cross_attention_kwargs=cross_attention_kwargs,
                # strength=0.8,             
                # guidance_scale=7.5      
            ).images[0]
            outputs.append(output)

            outputs = np.concatenate([np.asarray(img) for img in outputs], axis=1)
            image_save_path = ospj(save_path, f"{content_image_name}")
            imageio.imsave(image_save_path, outputs)


if __name__ == "__main__":
    opt = get_options()
        
    style_transfer(opt)
