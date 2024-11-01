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
import os

sys.path.append(os.path.dirname(sys.path[0]))
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from transformers import CLIPProcessor, CLIPModel

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


def main(prompt = '', content_dir = '', style_dir='',ddim_steps = 50,strength = 0.5, model = None, seed=42, save_path=None):
    ddim_eta=0.0
    n_iter=1
    C=4
    f=8
    n_samples=1
    n_rows=0
    scale=10.0
    
    precision="autocast"
    seed_everything(seed)

    batch_size = n_samples
    n_rows = n_rows if n_rows > 0 else batch_size
    data = [batch_size * [prompt]]

    style_img_path = os.path.join(style_dir, os.listdir(style_dir)[0])
    style_image = load_img(style_img_path).to(device)
    style_image = repeat(style_image, '1 ... -> b ...', b=batch_size)
    style_latent = model.get_first_stage_encoding(model.encode_first_stage(style_image))  # move to latent space

    content_file_name =  content_dir.split('/')[-1]
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
                            uc = model.get_learned_conditioning(batch_size * [""], style_image)
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)

                        c= model.get_learned_conditioning(prompts, style_image)

                        # img2img

                        # stochastic encode
                        # z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device))

                        # stochastic inversion
                        t_enc = int(strength * 1000)
                        x_noisy = model.q_sample(x_start=init_latent, t=torch.tensor([t_enc]*batch_size).to(device))
                        model_output = model.apply_model(x_noisy, torch.tensor([t_enc]*batch_size).to(device), c)
                        z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device),\
                                                          noise = model_output, use_original_steps = True)
            
                        t_enc = int(strength * ddim_steps)
                        samples = sampler.decode(z_enc, c, t_enc, 
                                                unconditional_guidance_scale=scale,
                                                 unconditional_conditioning=uc,)
                        # print(z_enc.shape, uc.shape, t_enc)

                        # txt2img
               #          noise  =torch.randn_like(content_latent)
               #          samples, intermediates =sampler.sample(ddim_steps,1,(4,512,512),c,verbose=False, eta=1.,x_T = noise,
               # unconditional_guidance_scale=scale,
               # unconditional_conditioning=uc,)

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
                # output.save(os.path.join(outpath, content_name+'-'+prompt+f'-{grid_count:04}.png'))
                
                output.save(os.path.join(save_path, content_file_name))
                # Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'grid-{grid_count:04}.png'))

                toc = time.time()
    return output

def gen_main(style_nums, content_dir, style_dir, save_path_dir, model):

    for style_num in style_nums.split(','):
        torch.cuda.empty_cache()
        
        #모델 호출
        style_emb_path = f'./trained_logs'
        style_emb_pt_path = os.path.join(style_emb_path, f'style_{str(style_num)}', 'checkpoints')
        
        model.embedding_manager.load(os.path.join(style_emb_pt_path, 'embeddings.pt'))
        model.cond_stage_model.mapper.load_state_dict(torch.load(os.path.join(style_emb_pt_path, 'Mapper.pt'), map_location='cpu'))
        model = model.to(device)
        model = model.eval()
        
        content_name = content_dir.split('/')[-1]
        save_path = os.path.join(save_path_dir, f'content_{content_name}', f'style_{style_num}')
        os.makedirs(save_path, exist_ok=True)
        
        for content_file in sorted(os.listdir(content_dir))[:1000]:
            content_path = os.path.join(content_dir, content_file)
                        
            main(prompt = '*', \
                content_dir = content_path, \
                style_dir = os.path.join(style_dir, style_num), \
                ddim_steps = 50,\
                strength = 0.5, \
                seed=42, \
                model = model, \
                save_path=save_path)


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
        "--content_path",
        type=str,
        default='../../dataset/content/custom5' #mscoco2017, face
    )
    parser.add_argument(
        "--style_path",
        type=str,
        default='../../dataset/style/wikiart' #wkiart,inst
    )
    parser.add_argument(
        "-sn",
        "--style_nums", 
        type=str,
        default='79', #wikiart-52,20,38,83,79 // #inst-1003,1004,1005
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=f'../../stylized_images/ArtBank'
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
    gen_main(opt.style_nums, opt.content_path, opt.style_path, opt.save_path, model)
