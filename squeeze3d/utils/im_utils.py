import os

import rembg
import torch
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from ..ldm.util import instantiate_from_config
from einops import rearrange
from ..instantmesh.utils.camera_util import (
    FOV_to_intrinsics,
    get_circular_camera_poses,
)
from ..instantmesh.utils.infer_util import (
    remove_background,
    resize_foreground,
)
from omegaconf import OmegaConf
import torchvision


def removebg(input_image):
    rembg_session = rembg.new_session()
    input_image = remove_background(input_image, rembg_session)
    input_image = resize_foreground(input_image, 0.85)
    return input_image


def load_zero123plus(infer_config, device):
    pipeline = DiffusionPipeline.from_pretrained(
        "sudo-ai/zero123plus-v1.2",
        custom_pipeline=os.path.join(os.path.dirname(__file__), "../", "zero123plus"),
        torch_dtype=torch.float16,
    )
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipeline.scheduler.config, timestep_spacing="trailing"
    )

    if os.path.exists(infer_config.unet_path):
        unet_ckpt_path = infer_config.unet_path
    else:
        unet_ckpt_path = hf_hub_download(
            repo_id="TencentARC/InstantMesh",
            filename="diffusion_pytorch_model.bin",
            repo_type="model",
        )
    state_dict = torch.load(unet_ckpt_path, map_location="cpu")
    pipeline.unet.load_state_dict(state_dict, strict=True)
    pipeline = pipeline.to(device)
    return pipeline


def get_render_cameras(
    batch_size=1, M=120, radius=4.0, elevation=20.0, is_flexicubes=False
):
    """
    Get the rendering camera parameters.
    """
    c2ws = get_circular_camera_poses(M=M, radius=radius, elevation=elevation)
    if is_flexicubes:
        cameras = torch.linalg.inv(c2ws)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    else:
        extrinsics = c2ws.flatten(-2)
        intrinsics = (
            FOV_to_intrinsics(30.0).unsqueeze(0).repeat(M, 1, 1).float().flatten(-2)
        )
        cameras = torch.cat([extrinsics, intrinsics], dim=-1)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1)
    return cameras


def render_frames(
    model, planes, render_cameras, render_size=512, chunk_size=1, is_flexicubes=False
):
    """
    Render frames from triplanes.
    """
    frames = []
    for i in tqdm(range(0, render_cameras.shape[1], chunk_size)):
        if is_flexicubes:
            frame = model.forward_geometry(
                planes,
                render_cameras[:, i : i + chunk_size],
                render_size=render_size,
            )["img"]
        else:
            frame = model.forward_synthesizer(
                planes,
                render_cameras[:, i : i + chunk_size],
                render_size=render_size,
            )["images_rgb"]
        frames.append(frame)

    frames = torch.cat(frames, dim=1)[0]  # we suppose batch size is always 1
    return frames


def load_instantmesh(config=None, device="cuda"):
    if config is None:
        config = os.path.join(
            os.path.dirname(__file__),
            "../",
            "../",
            "weights",
            "instantmesh",
            "instantmesh-large.yaml",
        )
    config = OmegaConf.load(config)
    model_config = config.model_config
    infer_config = config.infer_config
    model = instantiate_from_config(model_config)
    model_ckpt_path = infer_config.model_path
    state_dict = torch.load(model_ckpt_path, map_location="cpu")["state_dict"]
    state_dict = {
        k[14:]: v for k, v in state_dict.items() if k.startswith("lrm_generator.")
    }
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.init_flexicubes_geometry(device, fovy=30.0)
    model = model.eval()
    return model


def encode_instantmesh(model, images, input_cameras):
    images = torchvision.transforms.v2.functional.resize(
        images, 320, interpolation=3, antialias=True
    ).clamp(0, 1)
    B = images.shape[0]
    image_feats = model.encoder(images, input_cameras)
    image_feats = rearrange(image_feats, "(b v) l d -> b (v l) d", b=B)
    return image_feats


def decode_im(model, latents):
    planes = model.transformer(latents)
    return planes
