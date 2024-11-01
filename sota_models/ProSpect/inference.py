"""make variations of input image"""

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
import torch.nn.functional as F

sys.path.append(os.path.dirname(sys.path[0]))
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from transformers import CLIPProcessor, CLIPModel


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#torch.cuda.set_device(device)
cpu=torch.device("cpu")

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
    image = image.resize((512,512), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.




def gen_main(model, style_nums, style_guidance, content_dir, save_path_dir):    
    
    torch.cuda.empty_cache()
    for style_num in style_nums.split(','):
    
        style_emb_path = f'./trained_logs'
        style_emb_pt_path = os.path.join(style_emb_path, f'style_{str(style_num)}', 'checkpoints', 'embeddings.pt')
        
        model.embedding_manager.load(style_emb_pt_path)
        model = model.to(device)
        
        content_name = content_dir.split('/')[-1]
        save_path = os.path.join(save_path_dir, 'content_'+content_name, f'style_{style_num}', f'wt_{str(style_guidance)}')
        os.makedirs(save_path, exist_ok=True)
        
        for content_file in sorted(os.listdir(content_dir))[:1000]:
            content_path = os.path.join(content_dir, content_file)
            
            save_name=content_path.split('/')[-1]
            stylized = main(prompt = '*', \
                        content_dir = content_path, \
                        ddim_steps = 50, \
                        strength = style_guidance, \
                        seed=41, \
                        height = 512, \
                        width = 768, \
                        prospect_words = ['*',  # 10 generation ends\ 
                                            '*',  # 9 \
                                            '*',  # 8 \
                                            '*',  # 7 \ 
                                            '*',  # 6 \ 
                                            '*',  # 5 \
                                            '*',  # 4 \
                                            '*',  # 3 \
                                            '*',  # 2 \
                                            '*',  # 1 generation starts\ 
                                            ], \
                        model = model,\
                        sampler=sampler
                        )
            stylized.save(os.path.join(save_path, save_name))
        
#img2img
def main(prompt = '', content_dir = None, ddim_steps = 50, strength = 0.5, ddim_eta=0.0, n_iter=1, C=4, f=8, n_rows=0, scale=10.0, \
         model = None, seed=42, prospect_words = None, n_samples=1, height=512, width=512, sampler=None):
    
    precision="autocast"
    seed_everything(seed)
    
    batch_size = n_samples
    n_rows = n_rows if n_rows > 0 else batch_size
    data = [batch_size * [prompt]]

    
    if content_dir is not None:
        content_image = load_img(content_dir).to(device)
        content_image = repeat(content_image, '1 ... -> b ...', b=batch_size)
        content_latent = model.get_first_stage_encoding(model.encode_first_stage(content_image))  # move to latent space

        init_latent = content_latent

    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)

    assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(strength * ddim_steps)
    print(f"target t_enc is {t_enc} steps")

    precision_scope = autocast if precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                for n in trange(n_iter, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        if scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        
                        c= model.get_learned_conditioning(prompts, prospect_words=prospect_words)         
                        

                        #img2img
                        z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device))
                        t_enc = int(strength * ddim_steps)
                        samples = sampler.decode(z_enc, c, t_enc, 
                                                unconditional_guidance_scale=scale,
                                                 unconditional_conditioning=uc,)

                        x_samples = model.decode_first_stage(samples)

                        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                        for x_sample in x_samples:
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        all_samples.append(x_samples)

                # additionally, save as grid
                grid = torch.stack(all_samples, 0)
                grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                grid = make_grid(grid, nrow=n_rows)

                # to image
                grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                output = Image.fromarray(grid.astype(np.uint8))

                toc = time.time()
    return output



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
        "--content_path",
        type=str,
        default='../../dataset/content/custom5' #custom, face, mscoco2014, mscoco2017
    )
    parser.add_argument(
        "--style_path",
        type=str,
        default='../dataset/style/inst' #wkiart,inst
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
        default='../../stylized_images'#ProSpect' 
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=f'../../resources/configs/ProSpect/stable-diffusion/v1-inference.yaml'
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
    gen_main(model, \
            style_nums=opt.style_nums, \
            style_guidance=opt.style_guidance, \
            content_dir=opt.content_path, \
            save_path_dir=opt.save_path)
    