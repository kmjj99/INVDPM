from our_functions import *
from PIL import Image
from torchvision.utils import save_image

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

    # latents = prepare_latents(latents=latents)

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

pipe = stable_diffusion_pipe(model_id='/data/alswo9912/One-Step-Inversion/models/SwiftBrush',solver_order=1)
# /data/alswo9912/One-Step-Inversion/models/SwiftBrush 
orig_image = Image.open("/data/alswo9912/gnochi_mirror.jpeg").convert("RGB")
prompt = "a cat sitting next to a mirror"
# prompt_embeds, _ = encode_prompt(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder, prompt=prompt, do_classifier_free_guidance=False)
# orig_noise = 

# orig_image, prompt, _ = generate(model_id='/data/alswo9912/One-Step-Inversion/models/SwiftBrush',
#                                  prompt=prompt, num_inference_steps=1, solver_order=1)
orig_image.save("results/x_0_cat.png")

recon_noise = exact_inversion(orig_image, 
                              prompt, 
                              model_id='/data/alswo9912/One-Step-Inversion/models/SwiftBrush',
                              test_num_inference_steps=2,
                              inv_order=1, pipe=pipe)

decoded = pipe.vae.decode(recon_noise / pipe.vae.config.scaling_factor, return_dict=False)[0]

save_image(decoded, os.path.join("results", "x_t_cat.png"), normalize=True, value_range=(-1, 1))

# img = sample(unet=pipe.unet,
#                 tokenizer=pipe.tokenizer,
#                 text_encoder=pipe.text_encoder,
#                 scheduler=pipe.scheduler,
#                 vae=pipe.vae,
#                 latents=recon_noise.clone(),
#                 prompt_embeds=prompt_embeds,
#                 guidance_scale=1.0,
#                 timesteps=[999],
#                 return_type="decoded",
#                 prev=False)
# save_image(img, os.path.join("results", "x_0_cat.png"), normalize=True, value_range=(-1, 1))

img,_,_ = generate(model_id='/data/alswo9912/One-Step-Inversion/models/SwiftBrush',
                   prompt=prompt, 
                           init_latents=recon_noise, 
                           num_inference_steps=1, 
                           solver_order=1, pipe=pipe)
img.save("results/x_0_hat_cat.png")


