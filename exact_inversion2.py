import torch
import random
import os
import sys

import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from torchvision.utils import save_image
from argparse import ArgumentParser
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.transforms import ToPILImage

from our_functions import *
from torchvision.utils import save_image

class COCODataset(Dataset):
    def __init__(self, csv_file, coco_path, transform=None, image_path=None):
        if image_path is None:
            if not os.path.exists(coco_path):
                coco_path = "/data/datasets/coco/val2014"
            self.filenames = []
            self.prompts = []
            with open(csv_file, "r") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    filename = line[:29]
                    prompt = line[30:].replace("\"", "")
                    self.filenames.append(filename)
                    self.prompts.append(prompt)
            self.coco_path = coco_path
            self.transform = transform
            assert len(self.filenames) == len(self.prompts)
        else:
            self.image_paths = [image_path]
            self.prompts = ["a cat"] 

    def __len__(self):  
        return len(self.filenames)
    
    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.coco_path, self.filenames[idx])).convert("RGB")
        # img.save(os.path.join(self.save_path, self.filenames[idx]))
        if self.transform:
            img = self.transform(img)
        return img, self.prompts[idx], self.filenames[idx]

def sample(unet, tokenizer, text_encoder, scheduler, vae, latents, prompt=None, guidance_scale=1.0, prompt_embeds=None, timesteps=[999], guidance_rescale=0.0, return_type="pil", prev=True):
    if prompt is not None and guidance_scale > 1.0:
        do_classifier_free_guidance = True
    else:
        do_classifier_free_guidance = False

    if prompt is None:
        prompt = ""
    
    if prompt_embeds is None:
        prompt_embeds, negative_prompt_embeds = encode_prompt(tokenizer=tokenizer, text_encoder=text_encoder, prompt=prompt, do_classifier_free_guidance=do_classifier_free_guidance)

    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([prompt_embeds, negative_prompt_embeds], dim=0)

    for t in timesteps:
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=prompt_embeds)[0]

        if do_classifier_free_guidance:
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2, dim=0)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            if guidance_rescale > 0:
                std_cond = noise_pred_cond.std(dim=list(range(1, noise_pred_cond.ndim)), keepdim=True)
                std_cfg = noise_pred.std(dim=list(range(1, noise_pred.ndim)), keepdim=True)

                noise_pred_rescaled = noise_pred * (std_cond / std_cfg)
                noise_pred = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_pred
        
        if prev:
            latents = step_prev(scheduler=scheduler, noise_pred=noise_pred, t=t, latents=latents)
        else:
            latents = step(scheduler=scheduler, noise_pred=noise_pred, t=t, latents=latents)

    if return_type == "pt":
        return latents
    elif return_type == "decoded":
        return vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
    elif return_type == "pil":
        decoded = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
        decoded = (decoded / 2 + 0.5).clamp(0, 1)
        decoded = decoded.cpu().permute(0, 2, 3, 1).float().numpy()
        if len(decoded) == 3:
            decoded = decoded[None, ...]
        decoded = (decoded * 255).astype(np.uint8)
        decoded = [Image.fromarray(decoded[i]) for i in range(len(decoded))]
        return decoded

def encode_prompt(tokenizer, text_encoder, prompt=None, do_classifier_free_guidance=False, num_images_per_prompt=1,):
    text_inputs = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").to("cuda")

    text_input_ids = text_inputs.input_ids
    untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").to("cuda").input_ids

    if untruncated_ids.shape[1] > text_input_ids.shape[1] and not torch.equal(text_input_ids, untruncated_ids):
        removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])

    prompt_embeds = text_encoder(text_input_ids.to("cuda"), attention_mask=None)[0]

    bs_embeds, seq_len, _ = prompt_embeds.shape

    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(bs_embeds * num_images_per_prompt, seq_len, -1)

    if do_classifier_free_guidance:
        negative_text_inputs = tokenizer("", padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").to("cuda")
        negative_prompt_embeds = text_encoder(negative_text_inputs.input_ids.to("cuda"), attention_mask=None)[0]
        negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        negative_prompt_embeds = negative_prompt_embeds.view(bs_embeds * num_images_per_prompt, seq_len, -1)
        
    else:
        negative_prompt_embeds = None

    return prompt_embeds, negative_prompt_embeds

def step(scheduler, noise_pred, t, latents):
    alpha_prod_t = scheduler.alphas_cumprod[t]
    beta_prod_t = 1 - alpha_prod_t

    # prediction_type == "epsilon" 
    pred_original_sample = (latents - beta_prod_t ** 0.5 * noise_pred) / alpha_prod_t ** 0.5

    return pred_original_sample 

def step_prev(scheduler, noise_pred, t, latents, t_prev=None):
    alpha_prod_t = scheduler.alphas_cumprod[t]
    alpha_prod_t_prev = scheduler.alphas_cumprod[0] if t_prev is None else scheduler.alphas_cumprod[t_prev]
    beta_prod_t = 1 - alpha_prod_t

    # prediction_type == "epsilon" 
    pred_original_sample = (latents - beta_prod_t ** 0.5 * noise_pred) / alpha_prod_t ** 0.5

    std_dev = 0 
    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev ** 2) ** 0.5 * noise_pred
    prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction

    return prev_sample 

def extra_parse_args():
    parser = ArgumentParser()
    parser.add_argument("--exp", type=str, required=True)
    parser.add_argument("--inversion", choices=["DDIM", "ExactDDIM"], default="ExactDDIM")
    parser.add_argument("--num_inference_steps", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    
    args = parser.parse_args()
    return args

def main(args):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    torch.set_grad_enabled(False)
    
    model_id = '/data/alswo9912/One-Step-Inversion/models/SwiftBrush'

    transform = transforms.Compose([transforms.Resize(512),
                                    transforms.CenterCrop((512, 512)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                    ])

    dataset = COCODataset(csv_file=f"/data/alswo9912/One-Step-Inversion/coco_val2014_5k.csv", coco_path="/local_datasets/coco/val2014", transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    pipe = stable_diffusion_pipe(model_id=model_id,solver_order=1)
    pipe.set_progress_bar_config(disable=True)
    vae = pipe.vae
    unet = pipe.unet
    scheduler = pipe.scheduler
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder

    vae.eval()
    unet.eval()
    text_encoder.eval()

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)

    num_inference_steps = args.num_inference_steps

    save_path = f"fake_images/{args.exp}/coco_5k"
    if os.path.exists(save_path) and len(os.listdir(save_path)) == 5000:
        print(f"Save path {save_path} already exists and is not empty, exiting...")
        exit()
    os.makedirs(save_path, exist_ok=True)
    start_time = datetime.now()

    for i, (imgs, prompts, filenames) in enumerate(dataloader):

        imgs = imgs.to("cuda")
        imgs = pipe.vae.encode(imgs).latent_dist.sample()
        prompt_embeds, _ = encode_prompt(pipe.tokenizer, pipe.text_encoder, prompts, do_classifier_free_guidance=False)
        z_t_hat = exact_inversion(imgs, prompts[0], model_id=model_id, test_num_inference_steps=args.num_inference_steps, inv_order=1, pipe=pipe)

        x_0_hat = sample(unet=unet,
                    tokenizer=tokenizer,
                    text_encoder=text_encoder,
                    scheduler=scheduler,
                    vae=vae,
                    latents=z_t_hat,
                    prompt_embeds=prompt_embeds,
                    return_type="decoded")
        
        save_image(x_0_hat, os.path.join(save_path, filenames[0]), normalize=True, value_range=(-1, 1))

        if i % 10 == 0:
            print(f"Generated {(i + 1) * args.batch_size} images, time elapsed: {datetime.now() - start_time}, time remaining: {(len(dataset) - (i + 1) * args.batch_size) * (datetime.now() - start_time) / ((i + 1) * args.batch_size)}")
            sys.stdout.flush()

if __name__ == "__main__":
    # args = parse_args()
    extra_args = extra_parse_args()
    main(extra_args)