import torch

import numpy as np

from PIL import Image

from argparse import ArgumentParser
from diffusers.models import UNet2DConditionModel
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
from typing import Any, Dict, List, Optional, Tuple, Union
from torch.optim.lr_scheduler import ReduceLROnPlateau

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="models/SwiftBrush")
    parser.add_argument("-r", "--rank", type=int, default=4)
    parser.add_argument("--init_lora_weights", type=str, default="gaussian")
    parser.add_argument("--iter", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--cond", action="store_true")
    parser.add_argument("--optimizer_type", type=str, default="adam")
    parser.add_argument("--num_test_images", type=int, default=4)
    parser.add_argument("-t", "--timestep", type=int, default=999)
    parser.add_argument("--print_freq", type=int, default=10)
    parser.add_argument("--save_freq", type=int, default=1000)
    parser.add_argument("--img_freq", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exp", type=str, default=None)
    parser.add_argument("--update_scheme", default="lora")
    parser.add_argument("--loss_recon", action="store_true")
    parser.add_argument("--loss_recon_lambda", type=float, default=1.0)
    parser.add_argument("--loss_recon_type", type=str, default="latent_mse")
    parser.add_argument("--loss_recon_low_freq_lambda", type=float, default=1.0)
    parser.add_argument("--loss_recon_high_freq_lambda", type=float, default=0.0)
    parser.add_argument("--loss_recon2", action="store_true")
    parser.add_argument("--loss_recon2_lambda", type=float, default=1.0)
    parser.add_argument("--loss_recon2_type", type=str, default="lpips")
    parser.add_argument("--loss_clip", action="store_true")
    parser.add_argument("--loss_clip_lambda", type=float, default=1.0)
    parser.add_argument("--loss_noise", action="store_true")
    parser.add_argument("--loss_noise_lambda", type=float, default=1.0)
    parser.add_argument("--loss_noise_type", type=str, default="mse")
    parser.add_argument("--step_original", action="store_true")
    parser.add_argument("--use_full_dataset", action="store_true")
    parser.add_argument("--train_timestep", action="store_true")
    parser.add_argument("--timestep_clipping", action="store_true")
    parser.add_argument("--timestep_lr", type=float, default=1e-4)
    parser.add_argument("--train_timestep_using_loss_noise", action="store_true")
    parser.add_argument("--backbone", choices=["SwiftBrush", "InstaFlow", "RectifiedFlow", "DMD2"], default="SwiftBrush")
    parser.add_argument("--lora_layers", type=str, default="ver1")
    parser.add_argument("--loss_clip_direction", action="store_true")
    parser.add_argument("--loss_clip_direction_lambda", type=float, default=1.0)
    parser.add_argument("--loss_cycle", action="store_true")
    parser.add_argument("--loss_cycle_lambda", type=float, default=1.0)
    parser.add_argument("--skip_connection", type=bool, default=True)
    parser.add_argument("--selective_ratio", type=float, default=1.0)
    parser.add_argument("--log_freq", action="store_true")
    parser.add_argument("--log_grad_norm", action="store_true")
    parser.add_argument("--prediction_type", type=str, default="x_T", choices=["x_T", "epsilon"])
    parser.add_argument("--noisy_sample", action="store_true")
    parser.add_argument("--loss_diffusion", action="store_true")
    parser.add_argument("--loss_diffusion_lambda", type=float, default=1.0)
    parser.add_argument("--loss_self_distillation", action="store_true")
    parser.add_argument("--loss_self_distillation_lambda", type=float, default=1.0)
    parser.add_argument("--loss_multi_distillation", action="store_true")
    parser.add_argument("--loss_multi_distillation_lambda", type=float, default=1.0)
    parser.add_argument("--loss_multi_distillation_use_direct", action="store_true")
    parser.add_argument("--loss_source_distillation", action="store_true")
    parser.add_argument("--loss_source_distillation_lambda", type=float, default=1.0)
    parser.add_argument("--loss_source_distillation_type", type=str, default="mse")
    parser.add_argument("--log_test_lpips", action="store_true")
    parser.add_argument("--log_criteria", action="store_true")
    parser.add_argument("--use_noise_dataset", action="store_true")
    parser.add_argument("--noise_dataset_length", type=int, default=100)
    parser.add_argument("--flow_models_epsilon_sign", type=str, default="positive", choices=["positive", "negative"])
    args = parser.parse_args()
    return args

def encode_prompt(tokenizer, text_encoder, prompt=None, do_classifier_free_guidance=False, num_images_per_prompt=1,):
    text_inputs = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").to("cuda")

    text_input_ids = text_inputs.input_ids
    untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").to("cuda").input_ids

    if untruncated_ids.shape[1] > text_input_ids.shape[1] and not torch.equal(text_input_ids, untruncated_ids):
        removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
        # print(f"The following part of your input was truncated because CLIP can only handle sequences up to {tokenizer.model_max_length} tokens: {removed_text}")
    

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
    
def prepare_latents(latents):
    if latents is None:
        latents = torch.randn(1, 4, 64, 64, device="cuda")

    return latents

def step(scheduler, noise_pred, t, latents):
    alpha_prod_t = scheduler.alphas_cumprod[t]
    beta_prod_t = 1 - alpha_prod_t

    # prediction_type == "epsilon" 
    pred_original_sample = (latents - beta_prod_t ** 0.5 * noise_pred) / alpha_prod_t ** 0.5

    return pred_original_sample # Since it is an one-step diffusion.

def step_prev(scheduler, noise_pred, t, latents, t_prev=None):
    alpha_prod_t = scheduler.alphas_cumprod[t]
    alpha_prod_t_prev = scheduler.alphas_cumprod[0] if t_prev is None else scheduler.alphas_cumprod[t_prev]
    beta_prod_t = 1 - alpha_prod_t

    # prediction_type == "epsilon" 
    pred_original_sample = (latents - beta_prod_t ** 0.5 * noise_pred) / alpha_prod_t ** 0.5

    std_dev = 0 # Since it is a deterministic sampling, we set eta to 0.
    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev ** 2) ** 0.5 * noise_pred
    prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction

    return prev_sample # Since it is an one-step diffusion.

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

    latents = prepare_latents(latents=latents)

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
    
def prepare_latents_flow(latents, scheduler):
    if latents is None:
        latents = torch.randn(1, 4, 64, 64, device="cuda")

    latents = latents * scheduler.init_noise_sigma
    return latents

def sample_flow(unet, tokenizer, text_encoder, scheduler, vae, latents, prompt=None, guidance_scale=1.0, prompt_embeds=None, timesteps=[999], guidance_rescale=0.0, return_type="pil", num_inference_steps=1, *args, **kwargs):
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

    latents = prepare_latents_flow(latents=latents, scheduler=scheduler)
    dt = 1.0 / num_inference_steps
    timesteps = [(1 - i / num_inference_steps) * 1000 for i in range(num_inference_steps)] 
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
        
        latents = latents + dt * noise_pred

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
    
def backward_euler(pipe, latents, prompt=None, guidance_scale=1.0, prompt_embeds=None, timesteps=[999], guidance_rescale=0.0, return_type="pil", prev=True):
    if prompt is not None and guidance_scale > 1.0:
        do_classifier_free_guidance = True
    else:
        do_classifier_free_guidance = False

    if prompt is None:
        prompt = ""
    
    if prompt_embeds is None:
        prompt_embeds, negative_prompt_embeds = encode_prompt(pipe=pipe, prompt=prompt, do_classifier_free_guidance=do_classifier_free_guidance)

    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([prompt_embeds, negative_prompt_embeds], dim=0)

    #latents = prepare_latents(pipe=pipe, latents=latents)

    t = 0 
    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

    s = 999
    # s = 0
    t = 0
    
    lambda_s, lambda_t = torch.tensor(-2.6820).to("cuda"), torch.tensor(3.5347).to("cuda")
    sigma_s, sigma_t = torch.tensor(0.9977).to("cuda"), torch.tensor(0.0292).to("cuda") 
    h = lambda_t - lambda_s
    alpha_s, alpha_t = torch.tensor(0.0683).to("cuda"), torch.tensor(0.9996).to("cuda") 
    phi_1 = torch.expm1(-h)
    
    # lambda_s, lambda_t = torch.tensor(3.5347).to("cuda"), torch.tensor(3.5347).to("cuda")
    # sigma_s, sigma_t = torch.tensor(0.0292).to("cuda"), torch.tensor(0.0292).to("cuda") 
    # h = lambda_t - lambda_s
    # alpha_s, alpha_t = torch.tensor(0.9996).to("cuda"), torch.tensor(0.9996).to("cuda") 
    # phi_1 = torch.expm1(-h)
    
    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents              
    
    # predict the noise residual
    noise_pred = pipe.unet(latent_model_input, 0, encoder_hidden_states=prompt_embeds).sample 

    model_s = noise_pred # (latents - sigma_t * noise_pred) / alpha_t
    x_t = latents
    
    latents = (sigma_s / sigma_t) * (latents + alpha_t * phi_1 * model_s)      

    latents = fixedpoint_correction(latents, s, t, x_t, pipe.unet, order=1, text_embeddings=prompt_embeds, guidance_scale=guidance_scale,
                                                    step_size=0.5, scheduler=True)

    return latents
    
def fixedpoint_correction(x, s, t, x_t, unet, r=None, order=1, n_iter=50, step_size=0.1, th=1e-3, 
                            model_s_output=None, model_r_output=None, text_embeddings=None, guidance_scale=3.0, 
                            scheduler=False, factor=0.5, patience=20, anchor=False, warmup=True, warmup_time=20):
    do_classifier_free_guidance = guidance_scale > 1.0
    if order==1:
        input = x.clone()
        original_step_size = step_size
        
        # step size scheduler, reduce when not improved
        if scheduler:
            step_scheduler = StepScheduler(current_lr=step_size, factor=factor, patience=patience)

        lambda_s, lambda_t = torch.tensor(-2.6820).to("cuda"), torch.tensor(3.5347).to("cuda")
        alpha_s, alpha_t = torch.tensor(0.0683).to("cuda"), torch.tensor(0.9996).to("cuda") 
        sigma_s, sigma_t = torch.tensor(0.9977).to("cuda"), torch.tensor(0.0292).to("cuda") 
        
        # lambda_s, lambda_t = torch.tensor(3.5347).to("cuda"), torch.tensor(3.5347).to("cuda")
        # sigma_s, sigma_t = torch.tensor(0.0292).to("cuda"), torch.tensor(0.0292).to("cuda") 
        # alpha_s, alpha_t = torch.tensor(0.9996).to("cuda"), torch.tensor(0.9996).to("cuda") 
        
        h = lambda_t - lambda_s
        phi_1 = torch.expm1(-h)

        for i in range(n_iter):
            # step size warmup
            if warmup:
                if i < warmup_time:
                    step_size = original_step_size * (i+1)/(warmup_time)
            
            latent_model_input = (torch.cat([input] * 2) if do_classifier_free_guidance else input)
            
            noise_pred = unet(latent_model_input , s, encoder_hidden_states=text_embeddings).sample                 
            model_output = noise_pred

            x_t_pred = (sigma_t / sigma_s) * input - (alpha_t * phi_1 ) * model_output

            loss = torch.nn.functional.mse_loss(x_t_pred, x_t, reduction='sum')
            
            if loss.item() < th:
                break                
            
            # forward step method
            input = input - step_size * (x_t_pred- x_t)

            if scheduler:
                step_size = step_scheduler.step(loss)

        return input    

def backward_euler(pipe, latents, prompt=None, guidance_scale=1.0, prompt_embeds=None, timesteps=[999], guidance_rescale=0.0, return_type="pil", prev=True):
    if prompt is not None and guidance_scale > 1.0:
        do_classifier_free_guidance = True
    else:
        do_classifier_free_guidance = False

    if prompt is None:
        prompt = ""
    
    if prompt_embeds is None:
        prompt_embeds, negative_prompt_embeds = encode_prompt(pipe=pipe, prompt=prompt, do_classifier_free_guidance=do_classifier_free_guidance)

    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([prompt_embeds, negative_prompt_embeds], dim=0)

    #latents = prepare_latents(pipe=pipe, latents=latents)

    t = 0 
    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

    s = 999
    # s = 0
    t = 0
    
    lambda_s, lambda_t = torch.tensor(-2.6820).to("cuda"), torch.tensor(3.5347).to("cuda")
    sigma_s, sigma_t = torch.tensor(0.9977).to("cuda"), torch.tensor(0.0292).to("cuda") 
    h = lambda_t - lambda_s
    alpha_s, alpha_t = torch.tensor(0.0683).to("cuda"), torch.tensor(0.9996).to("cuda") 
    phi_1 = torch.expm1(-h)
    
    # lambda_s, lambda_t = torch.tensor(3.5347).to("cuda"), torch.tensor(3.5347).to("cuda")
    # sigma_s, sigma_t = torch.tensor(0.0292).to("cuda"), torch.tensor(0.0292).to("cuda") 
    # h = lambda_t - lambda_s
    # alpha_s, alpha_t = torch.tensor(0.9996).to("cuda"), torch.tensor(0.9996).to("cuda") 
    # phi_1 = torch.expm1(-h)
    
    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents              
    
    # predict the noise residual
    noise_pred = pipe.unet(latent_model_input, 0, encoder_hidden_states=prompt_embeds).sample 

    model_s = noise_pred # (latents - sigma_t * noise_pred) / alpha_t
    x_t = latents
    
    latents = (sigma_s / sigma_t) * (latents + alpha_t * phi_1 * model_s)      

    latents = fixedpoint_correction(latents, s, t, x_t, pipe.unet, order=1, text_embeddings=prompt_embeds, guidance_scale=guidance_scale,
                                                    step_size=0.5, scheduler=True)

    return latents
    
class StepScheduler(ReduceLROnPlateau):
    def __init__(self, mode='min', current_lr=0, factor=0.1, patience=10,
                 threshold=1e-4, threshold_mode='rel', cooldown=0,
                 min_lr=0, eps=1e-8, verbose=False):
        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0.')
        self.factor = factor
        if current_lr == 0:
            raise ValueError('Step size cannot be 0')

        self.min_lr = min_lr
        self.current_lr = current_lr
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.eps = eps
        self.last_epoch = 0
        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)
        self._reset()

    def step(self, metrics, epoch=None):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch + 1
        else:
            import warnings
            warnings.warn("EPOCH_DEPRECATION_WARNING", UserWarning)
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

        return self.current_lr

    def _reduce_lr(self, epoch):
        old_lr = self.current_lr
        new_lr = max(self.current_lr * self.factor, self.min_lr)
        if old_lr - new_lr > self.eps:
            self.current_lr = new_lr
            if self.verbose:
                epoch_str = ("%.2f" if isinstance(epoch, float) else
                            "%.5d") % epoch
                print('Epoch {}: reducing learning rate'
                        ' to {:.4e}.'.format(epoch_str,new_lr))

class UNet2DConditionModelWithSkip(UNet2DConditionModel):
    def __init__(self, *args, **kwargs):
        self.skip = kwargs.pop("skip", True) 
        super().__init__(*args, **kwargs)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet2DConditionOutput, Tuple]:

        default_overall_up_factor = 2**self.num_upsamplers
        forward_upsample_size = False
        upsample_size = None

        for dim in sample.shape[-2:]:
            if dim % default_overall_up_factor != 0:
                forward_upsample_size = True
                break

        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        if encoder_attention_mask is not None:
            encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        t_emb = self.get_time_embed(sample=sample, timestep=timestep)
        emb = self.time_embedding(t_emb, timestep_cond)

        class_emb = self.get_class_embed(sample=sample, class_labels=class_labels)
        if class_emb is not None:
            if self.config.class_embeddings_concat:
                emb = torch.cat([emb, class_emb], dim=-1)
            else:
                emb = emb + class_emb

        aug_emb = self.get_aug_embed(
            emb=emb, encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
        )
        if self.config.addition_embed_type == "image_hint":
            aug_emb, hint = aug_emb
            sample = torch.cat([sample, hint], dim=1)

        emb = emb + aug_emb if aug_emb is not None else emb

        if self.time_embed_act is not None:
            emb = self.time_embed_act(emb)

        encoder_hidden_states = self.process_encoder_hidden_states(
            encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
        )

        # 2. pre-process
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid
        if self.mid_block is not None:
            if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
                sample = self.mid_block(
                    sample,
                    emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = self.mid_block(sample, emb)

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(upsample_block.resnets)]

            if not self.skip:
                res_samples = tuple(torch.zeros_like(res) for res in res_samples)

            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                )

        # 6. post-process
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if not return_dict:
            return (sample,)

        return UNet2DConditionOutput(sample=sample)
