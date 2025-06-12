import argparse
import os
import torch
from tqdm import tqdm
from datasets import load_from_disk, load_dataset
from accelerate.utils import set_seed

import os
import random

import numpy as np
import torch


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


import sys

sys.path.append(
    os.path.join(
        os.path.dirname(__file__), "../", "Pointnet_Pointnet2_pytorch", "models"
    )
)
sys.path.append(os.path.dirname(__file__))

from inference import save_as_ply
from default_config import cfg as config
from models.lion import LION
from utils.vis_helper import plot_points


def process_and_save_dataset(dataset, output_dir, lion, device, split_name):
    """Process a dataset split and save PLY files"""
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for i, data_sample in enumerate(
            tqdm(dataset, desc=f"Processing {split_name} set")
        ):
            # Get the inputs from the dataset
            latent_vector = data_sample["inputs"]

            # Split the vector into z_global and z_local parts
            # z_global has 128 elements, z_local has 8192 elements
            z_global_flat = latent_vector[:128]
            z_local_flat = latent_vector[128:]

            # Reshape to the expected dimensions
            z_global = torch.tensor(z_global_flat, device=device).reshape(1, 128, 1, 1)
            z_local = torch.tensor(z_local_flat, device=device).reshape(1, 8192, 1, 1)

            # Create sampled_list as expected by the VAE
            sampled_list = [z_global, z_local]

            # Generate point clouds using the VAE's sample function
            points = lion.vae.sample(num_samples=1, decomposed_eps=sampled_list)

            # Get the generated points
            points_np = points.squeeze(0).cpu().numpy()

            # Save as PLY file
            ply_path = os.path.join(output_dir, f"{i}.ply")
            save_as_ply(points_np, ply_path)

            if i % 10 == 0:
                print(f"Saved {ply_path}")


def load_dataset_and_save_ply(
    dataset_path, output_dir, device, from_hub=False, repo_id=None, seed=42
):
    # Load dataset
    if from_hub:
        dataset = load_dataset(repo_id)[
            "train"
        ]  # Assuming the dataset has a 'train' split
    else:
        dataset = load_from_disk(dataset_path)

    # Create train/test split using the built-in method
    train_dataset, val_dataset = dataset.train_test_split(0.1, seed=seed).values()

    print(f"Train set size: {len(train_dataset)}")
    print(f"Test set size: {len(val_dataset)}")

    # Create output directories
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")

    # Load LION model
    checkpoint_dir = "/home/rishit/streaming-nerf/weights/lion_ckpt/unconditional/"
    category = "all55"
    model_path = os.path.join(
        checkpoint_dir, category, "checkpoints", "epoch_10999_iters_2100999.pt"
    )
    config_path = os.path.join(checkpoint_dir, category, "cfg.yml")
    device = torch.device(device)

    config.merge_from_file(config_path)
    lion = LION(config)
    lion.load_model(model_path)

    process_and_save_dataset(val_dataset, test_dir, lion, device, "test")
    process_and_save_dataset(train_dataset, train_dir, lion, device, "train")


def main():
    seed_all(3407)
    set_seed(3407)
    parser = argparse.ArgumentParser(
        description="Load dataset and save PLY files from LION model outputs"
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="latents",
        help="Path to the dataset on disk",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ply_outputs",
        help="Directory to save PLY files",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train-test split",
    )
    parser.add_argument(
        "--from_hub",
        action="store_true",
        help="Load dataset from Hugging Face Hub",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        help="Repository ID to load dataset from",
    )

    args = parser.parse_args()

    load_dataset_and_save_ply(
        args.dataset_path,
        args.output_dir,
        args.device,
        args.from_hub,
        args.repo_id,
        args.seed,
    )


if __name__ == "__main__":
    main()
