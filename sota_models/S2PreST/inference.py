import os
from tqdm import tqdm
import argparse, os, sys, glob
import PIL
import torch
import torch.nn as nn
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext
import time
from pytorch_lightning import seed_everything

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from transformers import CLIPProcessor, CLIPModel

from canny import save_canny


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.to(device)
    model.eval()
    return model


def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((512, 512), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.


def gen_main(style_nums, style_guidance, type, content_dir, style_path, save_path, model):
    
    #실행
    for style_num in style_nums.split(','):
        torch.cuda.empty_cache()
    
        if type == 's2prest':
            from modules.s2prest import main
            type = 'S2PreST'
        elif type == 'no_canny_1':
            from modules.no_canny_1 import main
            type = 'S2PreST_NoCanny1'
        elif type == 'no_canny_2':
            from modules.no_canny_2 import main
            type = 'S2PreST_NoCanny2'
        elif type == 'no_canny_3':
            from modules.no_canny_3 import main
            type = 'S2PreST_NoCanny3'
        elif type == 'no_reference':
            from modules.no_reference import main
            type = 'S2PreST_NoReference'
        elif type == 'inst':
            from modules.inst import main
            type = 'InST'
        elif type == 'mix_reference':
            from modules.s2prest_mix_reference import main
            type = 'S2PreST_MixReference'
        else:
            return
        
        style_emb_path = f'./trained_logs'
        style_emb_pt_path = os.path.join(style_emb_path, f'style_{str(style_num)}', 'checkpoints', 'embeddings.pt')
        
        model.embedding_manager.load(style_emb_pt_path)
        model = model.to(device)
        
        content_name = content_dir.split('/')[-1]
        
        for content_file in sorted(os.listdir(content_dir))[:1000]:
            content_path = os.path.join(content_dir, content_file)
                
            if type == 'InST':
                canny_path = None 
            else: # 's2prest','no_canny_1','no_canny_2','no_canny_3','no_reference'
                savepath_ca=f'./canny/{type}/content_{content_name}/style_{style_num}-wt_{str(style_guidance)}'
                canny_path = save_canny(content_path, savepath_ca)

            main(prompt = '*', \
                content_dir = content_path, \
                style_dir = os.path.join(style_path, style_num),
                canny_dir=canny_path, \
                ddim_steps = 50, \
                strength = style_guidance, \
                seed=42, \
                model = model,
                sampler = sampler,
                save_path = os.path.join(save_path, type, f'content_{content_name}', f'style_{str(style_num)}', f'wt_{str(style_guidance)}')
                )


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-sg",
        "--style_guidance",
        type=float,
        default=0.4,
    )
    parser.add_argument(
        "-t",
        "--type",
        type=str,
        default='inst' #s2prest, no_canny_1, no_canny_2, no_canny_3, no_reference, inst, s2prest_mix_reference
    )
    parser.add_argument(
        "--content_path",
        type=str,
        default='../../dataset/content/custom5' #mscoco2017, face
    )
    parser.add_argument(
        "--style_path",
        type=str,
        default='../../dataset/style/inst' #wkiart,inst
    )
    parser.add_argument(
        "-sn",
        "--style_nums", 
        type=str,
        default='1003,1004,1005', #wikiart-52,20,38,83,79 // #inst-1003,1004,1005
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=f'../../stylized_images'
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=f'../../resources/configs/InST/stable-diffusion/v1-inference.yaml'
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=f'../../resources/models/sd-v1-4/sd-v1-4.ckpt'
    )
    return parser
           
                          
if __name__ == "__main__":
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    
    #모델 호출
    config = OmegaConf.load(f"{opt.config_path}")
    model = load_model_from_config(config, f"{opt.ckpt_path}")
    sampler = DDIMSampler(model)
    
    #실행
    gen_main(opt.style_nums, opt.style_guidance, opt.type, opt.content_path, opt.style_path, opt.save_path, model)

