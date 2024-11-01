import os
import subprocess
import torch

style_start=None
style_end=None
style_num_target=[9] #not continual

if style_start and style_end:
    style_range = range(style_start, style_end + 1)
else:
    style_range = sorted(style_num_target)

for i in style_range:
    torch.cuda.empty_cache()
    style_num = i
    style_root = f"../dataset/style/wikiart/{str(style_num)}"
    
    if os.path.isdir(style_root):
        
        command = [
            "python", "main.py", 
            "--base", "configs/stable-diffusion/v1-finetune.yaml",
            "-t", 
            "--actual_resume", "./models/sd/sd-v1-4.ckpt", 
            "-n", f"style_{style_num}", 
            "--gpus", "0,", 
            "--no-test",
            "--data_root", style_root
        ]

        print("Running command:", " ".join(command))
        
        try:
            subprocess.run(command, check=True)
            print(f"Successfully ran command for folder: {style_root}")
            torch.cuda.empty_cache()
        except subprocess.CalledProcessError as e:
            print(f"Error running command for folder: {style_root}\nError: {e}")
    else:
        print(f"Directory does not exist: {style_root}")
