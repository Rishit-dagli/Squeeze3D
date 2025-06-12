import argparse
import os
from glob import glob
import torch
import numpy as np
from tqdm import tqdm
from datasets import Dataset, DatasetDict, concatenate_datasets
import torch.nn.functional as F

from squeeze3d.utils.nerfmae_utils import load_model, encode_to_latent, load_data
from squeeze3d.utils.utils import seed_all

seed_all(3407)


def create_nerf_dataset(
    nerf_dir,
    model_path,
    output_file,
    subset=None,
    num_samples=None,
    upload_to_hub=False,
    repo_id=None,
    token=None,
    backbone_type="swin_s",
    chunk_size=500,
):
    """
    Create a dataset of NeRF latents from .npz files.

    Args:
        nerf_dir (str): Directory containing NeRF .npz files
        model_path (str): Path to the pretrained NeRF-MAE model
        output_file (str): Path to save the dataset
        subset (str): Subset to use ('normal', '3dfront', 'ai')
        num_samples (int): Maximum number of samples to process
        upload_to_hub (bool): Whether to upload to HuggingFace Hub
        repo_id (str): Repository ID for HuggingFace Hub
        token (str): HuggingFace API token
        backbone_type (str): Type of Swin Transformer backbone
        chunk_size (int): Size of chunks to process and save
    """
    # Load the model
    print(f"Loading model from {model_path}")
    model, device = load_model(model_path, backbone_type=backbone_type)

    # Find all .npz files in the directory and subdirectories
    npz_files = glob(os.path.join(nerf_dir, "**/*.npz"), recursive=True)
    print(f"Found {len(npz_files)} .npz files")

    # Filter files based on subset
    if subset:
        if subset == "normal":
            # Exclude files starting with 3dfront_ or ai_
            npz_files = [
                f
                for f in npz_files
                if not (
                    os.path.basename(f).startswith("3dfront_")
                    or os.path.basename(f).startswith("ai_")
                )
            ]
        elif subset == "3dfront":
            # Only include files starting with 3dfront_
            npz_files = [
                f for f in npz_files if os.path.basename(f).startswith("3dfront_")
            ]
        elif subset == "ai":
            # Only include files starting with ai_
            npz_files = [f for f in npz_files if os.path.basename(f).startswith("ai_")]

    # Limit the number of samples if specified
    if num_samples and num_samples > 0:
        npz_files = npz_files[:num_samples]

    print(f"Processing {len(npz_files)} files after filtering")

    # Process each file and encode to latent space
    encoded_latents = []

    for file_path in tqdm(npz_files, desc="Encoding NeRFs"):
        try:
            nerf_field = load_data(file_path)

            # Encode to latent space
            latent_representation = encode_to_latent(nerf_field, model, device)

            # Store the latent representation (already has batch dimension)
            encoded_latents.append(latent_representation.cpu().numpy())

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

    # Define a function to chunk a list into smaller lists
    def chunk_list(data_list, chunk_size):
        for i in range(0, len(data_list), chunk_size):
            yield data_list[i : i + chunk_size]

    # Process the data in chunks to avoid memory issues
    if len(encoded_latents) > chunk_size:
        print(f"Creating dataset in chunks of size {chunk_size}")
        chunk_datasets = []

        # Process data in chunks
        for i, chunk in enumerate(chunk_list(encoded_latents, chunk_size)):
            print(
                f"Processing chunk {i+1}/{(len(encoded_latents) + chunk_size - 1) // chunk_size}"
            )
            chunk_dataset = Dataset.from_dict({"latent": chunk})
            chunk_datasets.append(chunk_dataset)

        # Concatenate all chunk datasets
        train_dataset = concatenate_datasets(chunk_datasets)
    else:
        # Create the dataset with only the encoded latents (small enough to process at once)
        train_dataset = Dataset.from_dict({"latent": encoded_latents})

    # Create a dataset dictionary with only the train split
    dataset_dict = DatasetDict({"train": train_dataset})

    # Save or upload dataset
    if upload_to_hub:
        if not repo_id or not token:
            raise ValueError("repo_id and token are required for uploading to Hub")
        dataset_dict.push_to_hub(repo_id, token=token)
        print(f"Dataset uploaded to Hugging Face Hub: {repo_id}")
    else:
        dataset_dict.save_to_disk(output_file)
        print(f"Dataset saved to disk: {output_file}")

    return dataset_dict


def main():
    parser = argparse.ArgumentParser(description="Create a dataset of NeRF latents")

    # Input and output options
    parser.add_argument(
        "--nerf_dir",
        type=str,
        default="/scratch/rishit/NeRF-MAE/pretrain/features/",
        help="Directory containing NeRF .npz files",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the pretrained NeRF-MAE model",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="nerf_latent_dataset",
        help="Output file for the dataset",
    )

    # Dataset filtering options
    parser.add_argument(
        "--subset",
        type=str,
        choices=["normal", "3dfront", "ai"],
        help="Subset of data to use",
    )
    parser.add_argument("--num_samples", type=int, help="Number of samples to process")

    # Model options
    parser.add_argument(
        "--backbone_type",
        type=str,
        default="swin_s",
        choices=["swin_t", "swin_s", "swin_b", "swin_l"],
        help="Type of Swin Transformer backbone",
    )

    # Upload options
    parser.add_argument(
        "--upload_to_hub",
        action="store_true",
        help="Upload dataset to Hugging Face Hub",
    )
    parser.add_argument(
        "--repo_id", type=str, help="Repository ID for Hugging Face Hub"
    )
    parser.add_argument("--token", type=str, help="Hugging Face API token")

    # Chunking options
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=100,
        help="Size of chunks to process and save",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.upload_to_hub and (not args.repo_id or not args.token):
        parser.error("--repo_id and --token are required when --upload_to_hub is set")

    create_nerf_dataset(
        nerf_dir=args.nerf_dir,
        model_path=args.model_path,
        output_file=args.output_file,
        subset=args.subset,
        num_samples=args.num_samples,
        upload_to_hub=args.upload_to_hub,
        repo_id=args.repo_id,
        token=args.token,
        backbone_type=args.backbone_type,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
