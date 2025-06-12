import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms

from ..ldm.models.diffusion.ddim import DDIMSampler
from ..ldm.util import instantiate_from_config, load_and_preprocess


def load_model(ckpt_path, config_path, device="cuda:1"):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)
    pl_sd = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(pl_sd["state_dict"], strict=False)
    model.to(device)
    model.eval()
    return model


def get_latents(model, input_image, device="cuda:1"):
    with torch.no_grad():
        input_im = input_image.resize((256, 256), Image.Resampling.LANCZOS)
        input_im = np.asarray(input_im, dtype=np.float32) / 255.0
        alpha = input_im[:, :, 3:4]
        white_im = np.ones_like(input_im)
        input_im = alpha * input_im + (1.0 - alpha) * white_im
        input_im = input_im[:, :, 0:3]
        input_im = input_im * 2 - 1
        input_im = transforms.ToTensor()(input_im).unsqueeze(0).to(device)
        latents = model.encode_first_stage(input_im)
    return latents


def generate_multiview_outputs(model, latents, num_views=4):
    outputs = []
    sampler = DDIMSampler(model)
    for _ in range(num_views):
        with torch.no_grad():
            output = model.decode_first_stage(latents)
            output = 255.0 * rearrange(output.cpu().numpy(), "c h w -> h w c")
            outputs.append(Image.fromarray(output.astype(np.uint8)))
    return outputs
