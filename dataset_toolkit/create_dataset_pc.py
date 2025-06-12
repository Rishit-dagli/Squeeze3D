import argparse
import io
import os
import random
import shutil
import tempfile

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from datasets import Dataset, concatenate_datasets

from squeeze3d.utils.utils import seed_all
from plyfile import PlyData
import importlib
import sys

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "../",
        "squeeze3d",
        "Pointnet_Pointnet2_pytorch",
        "models",
    )
)
sys.path.append(os.path.join(os.path.dirname(__file__), "../", "squeeze3d", "LION"))

from pointnet_inference import (
    load_ply,
    pc_normalize,
    process_point_cloud,
    get_model_and_labels,
)
from inference import save_as_obj, save_as_ply
from default_config import cfg as config
from models.lion import LION
from utils.vis_helper import plot_points

seed_all(3407)


@torch.no_grad()
def sample(self, num_samples=10, clip_feat=None, save_img=False):
    self.scheduler.set_timesteps(1000, device="cuda")
    timesteps = self.scheduler.timesteps
    latent_shape = self.vae.latent_shape()
    global_prior, local_prior = self.priors[0], self.priors[1]
    assert not local_prior.mixed_prediction and not global_prior.mixed_prediction
    sampled_list = []
    output_dict = {}

    x_T_shape = [num_samples] + latent_shape[0]
    x_noisy = torch.randn(size=x_T_shape, device="cuda")
    condition_input = None
    for i, t in enumerate(timesteps):
        t_tensor = torch.ones(num_samples, dtype=torch.int64, device="cuda") * (t + 1)
        noise_pred = global_prior(
            x=x_noisy,
            t=t_tensor.float(),
            condition_input=condition_input,
            clip_feat=clip_feat,
        )
        x_noisy = self.scheduler.step(noise_pred, t, x_noisy).prev_sample
    sampled_list.append(x_noisy)
    output_dict["z_global"] = x_noisy
    # print(f"z_global: {x_noisy.shape}")

    condition_input = x_noisy
    condition_input = self.vae.global2style(condition_input)

    x_T_shape = [num_samples] + latent_shape[1]
    x_noisy = torch.randn(size=x_T_shape, device="cuda")

    for i, t in enumerate(timesteps):
        t_tensor = torch.ones(num_samples, dtype=torch.int64, device="cuda") * (t + 1)
        noise_pred = local_prior(
            x=x_noisy,
            t=t_tensor.float(),
            condition_input=condition_input,
            clip_feat=clip_feat,
        )
        x_noisy = self.scheduler.step(noise_pred, t, x_noisy).prev_sample
    sampled_list.append(x_noisy)
    output_dict["z_local"] = x_noisy
    # print(f"z_local: {x_noisy.shape}")
    # print(f"sampled_list: {len(sampled_list)}")

    output = self.vae.sample(num_samples=num_samples, decomposed_eps=sampled_list)
    if save_img:
        out_name = plot_points(output, "/tmp/tmp.png")
        print(f"INFO save plot image at {out_name}")
    output_dict["points"] = output
    return output_dict, sampled_list


def create_dataset(
    num_samples,
    device,
    output_file,
    upload_to_hub,
    repo_id,
    token,
):
    checkpoint_dir = os.path.join(
        os.path.dirname(__file__), "../", "weights", "lion_ckpt", "unconditional"
    )
    category = "all55"
    model_path = os.path.join(
        checkpoint_dir, category, "checkpoints", "epoch_10999_iters_2100999.pt"
    )
    config_path = os.path.join(checkpoint_dir, category, "cfg.yml")
    device = torch.device(device)

    config.merge_from_file(config_path)
    lion = LION(config)
    lion.load_model(model_path)

    classifier, label_map = get_model_and_labels(
        "pointnet2_cls_ssg", 40, normal_channel=False
    )
    checkpoint = torch.load(
        os.path.join(
            os.path.dirname(__file__),
            "../",
            "squeeze3d",
            "Pointnet_Pointnet2_pytorch",
            "log",
            "classification",
            "pointnet2_ssg_wo_normals",
            "checkpoints",
            "best_model.pth",
        )
    )
    classifier.load_state_dict(checkpoint["model_state_dict"])
    classifier.eval()
    classifier = classifier.to(device)

    inputs = []
    outputs = []

    for i in tqdm(range(num_samples)):
        output, sampled_list = sample(lion, 1)
        pts = output["points"].squeeze(0)
        input_latents = torch.cat(
            [sampled_list[0].squeeze(0), sampled_list[1].squeeze(0)]
        )
        inputs.append(input_latents.cpu().numpy())

        pts = pts.cpu().numpy()
        processed_pc = process_point_cloud(
            pts, int(pts.shape[0]), use_normals=False
        ).to(device)
        with torch.no_grad():
            pred, _ = classifier(processed_pc)
        outputs.append(pred.squeeze(0).cpu().numpy())

    def chunk_dict(data_dict, chunk_size):
        length = len(next(iter(data_dict.values())))
        for i in range(0, length, chunk_size):
            chunk = {k: v[i : i + chunk_size] for k, v in data_dict.items()}
            yield chunk

    data_dict = {"inputs": inputs, "outputs": outputs}
    chunk_size = 500
    chunks = chunk_dict(data_dict, chunk_size)
    chunk_datasets = []
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i}")
        chunk_dataset = Dataset.from_dict(chunk)
        chunk_datasets.append(chunk_dataset)
    dataset = concatenate_datasets(chunk_datasets)

    if upload_to_hub:
        dataset.push_to_hub(repo_id, token=token)
        print(f"Dataset uploaded to Hugging Face Hub: {repo_id}")
    else:
        dataset.save_to_disk(output_file)
        print(f"Dataset saved to disk: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Create a HuggingFace dataset of latents"
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="latents",
        help="Output file name",
    )
    parser.add_argument(
        "--upload_to_hub",
        action="store_true",
        help="Upload dataset to Hugging Face Hub",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        help="Repository ID to upload dataset to",
    )
    parser.add_argument(
        "--token",
        type=str,
        help="Hugging Face API token",
    )

    args = parser.parse_args()

    create_dataset(
        args.num_samples,
        args.device,
        args.output_file,
        args.upload_to_hub,
        args.repo_id,
        args.token,
    )


if __name__ == "__main__":
    main()
