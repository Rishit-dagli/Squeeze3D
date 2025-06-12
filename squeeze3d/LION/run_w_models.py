import argparse
import json
import os
from statistics import mean
import time
import torch
from tqdm import tqdm
from datasets import load_from_disk, load_dataset
from accelerate.utils import set_seed
import torch.nn as nn
import torch.nn.functional as F
import torch
from einops import rearrange, repeat
import math
from typing import Callable, Tuple, List, Union, Optional
import os
import random
import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file


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


class LTpc(nn.Module):
    def __init__(
        self,
        input_shape=(1024,),
        output_size: int = 8320,
        hidden_size: int = 2048,
        num_layers: int = 12,
        dropout_prob: float = 0.0,
        activation=F.gelu,
    ):
        super().__init__()

        if len(input_shape) == 3:
            input_size = input_shape[0] * input_shape[1] * input_shape[2]
        elif len(input_shape) == 2:
            input_size = input_shape[0] * input_shape[1]
        else:
            input_size = input_shape[0]

        self.flatten = nn.Flatten()
        self.activation = activation
        self.num_layers = num_layers

        self.first_layer = nn.Linear(input_size, hidden_size)

        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))

        self.layer_norm = nn.LayerNorm(hidden_size)

        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0 else nn.Identity()

        self.output_layer = nn.Linear(hidden_size, output_size)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)

        x = self.first_layer(x)

        x = self.layer_norm(x)
        residual = x

        x = self.activation(x)
        x = self.dropout(x)

        for i, layer in enumerate(self.hidden_layers):
            layer_input = x
            x = layer(x)

            if i == 0:
                x = x + residual
            elif i % 4 == 0:
                x = x + layer_input

            x = self.activation(x)
            x = self.dropout(x)

        x = self.output_layer(x)

        return x


def process_model_config(
    train_dataset,
    test_dataset,
    model_config,
    model_name,
    safetensors_path,
    lion,
    device,
    base_output_dir,
):
    """Process a single model configuration"""
    model = LTpc(**model_config).to(device)

    # Load weights from safetensors
    state_dict = load_file(safetensors_path)
    model.load_state_dict(state_dict)
    model.eval()

    # Create output directories
    model_output_dir = os.path.join(base_output_dir, model_name)
    train_dir = os.path.join(model_output_dir, "train")
    test_dir = os.path.join(model_output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Track timing information
    model_times = []
    vae_times = []

    # Process test dataset
    process_dataset_split(
        test_dataset, test_dir, model, lion, device, "test", model_times, vae_times
    )

    # Process train dataset
    process_dataset_split(
        train_dataset, train_dir, model, lion, device, "train", model_times, vae_times
    )

    # Calculate average times
    avg_model_time = mean(model_times) if model_times else 0
    avg_vae_time = mean(vae_times) if vae_times else 0

    # Save timing information
    with open(os.path.join(model_output_dir, "timing.txt"), "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Average model inference time: {avg_model_time:.2f} ms\n")
        f.write(f"Average VAE sampling time: {avg_vae_time:.2f} ms\n")
        f.write(f"Total average time: {avg_model_time + avg_vae_time:.2f} ms\n")

    return {
        "model_name": model_name,
        "avg_model_time": avg_model_time,
        "avg_vae_time": avg_vae_time,
        "total_time": avg_model_time + avg_vae_time,
    }


@torch.no_grad()
@torch.inference_mode()
def process_dataset_split(
    dataset, output_dir, model, lion, device, split_name, model_times, vae_times
):
    """Process a dataset split and save PLY files"""

    for i, data_sample in enumerate(tqdm(dataset, desc=f"Processing {split_name} set")):
        # Get the classifier outputs from the dataset
        classifier_outputs = data_sample["outputs"]
        outputs_tensor = torch.tensor(classifier_outputs, device=device).unsqueeze(
            0
        )  # Add batch dimension

        # Time the model inference
        torch.cuda.synchronize()
        start_time = time.time()

        # Pass outputs to the model to get latents
        latents = model(outputs_tensor)

        torch.cuda.synchronize()
        end_time = time.time()
        model_times.append((end_time - start_time) * 1000)  # Convert to milliseconds

        # Extract and reshape latents
        latent_vector = latents.squeeze(0).cpu().numpy()  # Shape should be 8320

        # Split the vector into z_global and z_local parts
        z_global_flat = latent_vector[:128]
        z_local_flat = latent_vector[128:]

        # Reshape to the expected dimensions
        z_global = torch.tensor(z_global_flat, device=device).reshape(1, 128, 1, 1)
        z_local = torch.tensor(z_local_flat, device=device).reshape(1, 8192, 1, 1)

        # Create sampled_list as expected by the VAE
        sampled_list = [z_global, z_local]

        # Time the VAE sampling
        torch.cuda.synchronize()
        start_time = time.time()

        # Generate point clouds using the VAE's sample function
        points = lion.vae.sample(num_samples=1, decomposed_eps=sampled_list)

        torch.cuda.synchronize()
        end_time = time.time()
        vae_times.append((end_time - start_time) * 1000)  # Convert to milliseconds

        # Get the generated points
        points_np = points.squeeze(0).cpu().numpy()

        # Save as PLY file
        ply_path = os.path.join(output_dir, f"{i}.ply")
        save_as_ply(points_np, ply_path)

        # if i % 10 == 0:
        #     print(f"Saved {ply_path}")


def load_dataset_and_process_models(
    dataset_path,
    model_configs_file,
    base_output_dir,
    device,
    from_hub=False,
    repo_id=None,
    seed=42,
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

    # Load LION model for VAE sampling
    checkpoint_dir = os.path.join(
        os.path.dirname(__file__), "../", "../", "weights", "lion_ckpt", "unconditional"
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

    # Create base output directory
    os.makedirs(base_output_dir, exist_ok=True)

    # Load model configurations
    with open(model_configs_file, "r") as f:
        model_configs = json.load(f)

    # Process each model configuration
    results = []
    for model_config in model_configs:
        result = process_model_config(
            train_dataset,
            val_dataset,
            model_config["config"],
            model_config["name"],
            model_config["safetensors_path"],
            lion,
            device,
            base_output_dir,
        )
        results.append(result)

    # Save summary of all models
    with open(os.path.join(base_output_dir, "summary.txt"), "w") as f:
        f.write("Summary of all models:\n\n")
        for result in results:
            f.write(f"Model: {result['model_name']}\n")
            f.write(
                f"Average model inference time: {result['avg_model_time']:.2f} ms\n"
            )
            f.write(f"Average VAE sampling time: {result['avg_vae_time']:.2f} ms\n")
            f.write(f"Total average time: {result['total_time']:.2f} ms\n")
            f.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Process multiple model configurations and save PLY files"
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="latents",
        help="Path to the dataset on disk",
    )
    parser.add_argument(
        "--model_configs_file",
        type=str,
        required=True,
        help="Path to JSON file containing model configurations",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/rishit/streaming-nerf/notebooks/fig_testing/pc",
        help="Base directory to save outputs",
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

    load_dataset_and_process_models(
        args.dataset_path,
        args.model_configs_file,
        args.output_dir,
        args.device,
        args.from_hub,
        args.repo_id,
        args.seed,
    )


if __name__ == "__main__":
    main()
