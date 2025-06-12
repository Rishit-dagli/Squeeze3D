import os
import sys
import torch
import numpy as np
import argparse
from pathlib import Path
import gc
from safetensors.torch import load_file

from squeeze3d.models.mlp import LTNeRFOrtho
from squeeze3d.utils.utils import seed_all
from squeeze3d.utils.nerfmae_utils import (
    load_model,
    encode_to_latent,
    load_data,
    decode_from_latent,
)


seed_all(3407)


def calculate_rf_size(rf_path):
    rf_path = Path(rf_path)
    if rf_path.exists():
        return rf_path.stat().st_size
    return 0


def encode_with_nerfmae(rf_path, model, device):

    nerf_field = load_data(rf_path)

    with torch.no_grad():
        latent_representation = encode_to_latent(nerf_field, model, device)
        encoded_latent = latent_representation.squeeze(0)

    return encoded_latent


def decode_with_nerfmae(compressed_output, model, device):

    compressed_output = compressed_output.unsqueeze(0)

    with torch.no_grad():
        reconstructed_field = decode_from_latent(compressed_output, model, device)

    return reconstructed_field


def save_radiance_field(field, output_path):

    if torch.is_tensor(field):
        field = field.cpu().numpy()
    np.savez_compressed(output_path, field=field)


def main():
    parser = argparse.ArgumentParser(
        description="Compress a radiance field using Squeeze3D with NeRF-MAE"
    )
    parser.add_argument(
        "rf_path", type=str, help="Path to input radiance field file (.npz)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output_rf.npz",
        help="Path for output radiance field",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--weights_dir",
        type=str,
        default="weights",
        help="Directory containing model weights",
    )

    parser.add_argument(
        "--squeeze3d_weights",
        type=str,
        default=None,
        help="Path to Squeeze3D model weights (overrides weights_dir)",
    )
    parser.add_argument(
        "--nerfmae_weights",
        type=str,
        default=None,
        help="Path to NeRF-MAE model weights",
    )
    parser.add_argument(
        "--backbone_type",
        type=str,
        default="swin_s",
        choices=["swin_t", "swin_s", "swin_b", "swin_l"],
        help="Type of Swin Transformer backbone",
    )

    args = parser.parse_args()

    args.output_path = os.path.abspath(args.output_path)

    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if args.device == "cuda":
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    original_size = calculate_rf_size(args.rf_path)

    device = torch.device(args.device)

    if args.nerfmae_weights:
        model_path = args.nerfmae_weights
    else:
        model_path = os.path.join(args.weights_dir, "nerfmae", "model.pth")

    if not os.path.exists(model_path):
        print(f"Error: NeRF-MAE weights not found at {model_path}")
        return

    nerfmae_model, _ = load_model(model_path, backbone_type=args.backbone_type)

    rf_latent = encode_with_nerfmae(args.rf_path, nerfmae_model, device)
    rf_size = rf_latent.numel() * 4

    rf_latent_cpu = rf_latent.cpu()
    del rf_latent
    torch.cuda.empty_cache()
    gc.collect()

    compression_model = LTNeRFOrtho(
        dropout=0.2,
    )

    if args.squeeze3d_weights:
        model_path = args.squeeze3d_weights
    else:
        model_path = os.path.join(args.weights_dir, "rf_nerfmae.safetensors")

    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}")
        return

    compression_model.load_state_dict(load_file(model_path))
    compression_model.to(device).eval()

    rf_latent_gpu = rf_latent_cpu.to(device)
    with torch.no_grad():
        compressed, b_matrix = compression_model(rf_latent_gpu.unsqueeze(0))

        b_matrix_squeezed = b_matrix.squeeze(0)
        compressed_size = b_matrix_squeezed.numel() * 4

    compressed_cpu = compressed.cpu()
    b_matrix_cpu = b_matrix.squeeze(0).cpu()
    del compression_model
    del rf_latent_gpu
    del compressed
    del b_matrix
    del b_matrix_squeezed
    torch.cuda.empty_cache()
    gc.collect()

    compressed_gpu = compressed_cpu.to(device)
    reconstructed_field = decode_with_nerfmae(
        compressed_gpu.squeeze(0), nerfmae_model, device
    )

    save_radiance_field(reconstructed_field, args.output_path)

    output_size = calculate_rf_size(args.output_path)

    compressed_path = args.output_path.replace(".npz", "_compressed.pt")
    torch.save(b_matrix_cpu, compressed_path)

    print(f"\nCompression ratio: {original_size / compressed_size:.2f}x")
    print(f"Compressed representation saved to: {compressed_path}")
    print(f"Reconstructed radiance field saved to: {args.output_path}")


if __name__ == "__main__":
    main()
