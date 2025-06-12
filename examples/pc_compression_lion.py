import os
import sys
import torch
import numpy as np
import argparse
from pathlib import Path
import gc
from plyfile import PlyData, PlyElement
from safetensors.torch import load_file


sys.path.append(
    os.path.join(
        os.path.dirname(__file__), "../squeeze3d/Pointnet_Pointnet2_pytorch/models"
    )
)
sys.path.append(os.path.join(os.path.dirname(__file__), "../squeeze3d/LION"))

from squeeze3d.models.mlp import LTOrtho, LTpcOrtho
from squeeze3d.utils.utils import seed_all
from pointnet_inference import (
    load_ply,
    pc_normalize,
    process_point_cloud,
    get_model_and_labels,
)
from default_config import cfg as config
from models.lion import LION
from inference import save_as_ply


seed_all(3407)


def calculate_pc_size(pc_path):
    pc_path = Path(pc_path)
    if pc_path.exists():
        return pc_path.stat().st_size
    return 0


def load_point_cloud(pc_path):
    plydata = PlyData.read(pc_path)
    vertices = plydata["vertex"]
    points = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
    return points


def encode_with_pointnet(pc_path, classifier, device):

    points = load_point_cloud(pc_path)
    points = pc_normalize(points)

    processed_pc = process_point_cloud(
        points, int(points.shape[0]), use_normals=False
    ).to(device)

    with torch.no_grad():
        pred, _ = classifier(processed_pc)
        encoded_latent = pred.squeeze(0)

    return encoded_latent, len(points)


def decode_with_lion(compressed_output, lion_model, device):

    latent_shapes = lion_model.vae.latent_shape()

    global_size = latent_shapes[0][0]
    local_size = latent_shapes[1][0]

    z_global = (
        compressed_output[:global_size].unsqueeze(0).reshape(1, global_size, 1, 1)
    )
    z_local = compressed_output[global_size:].unsqueeze(0)

    if z_local.shape[1] < local_size:
        padding = local_size - z_local.shape[1]
        z_local = torch.nn.functional.pad(
            z_local, (0, padding), mode="constant", value=0
        )

    z_local = z_local.reshape(1, local_size, 1, 1)

    sampled_list = [z_global, z_local]

    with torch.no_grad():
        output = lion_model.vae.sample(num_samples=1, decomposed_eps=sampled_list)
        points = output.squeeze(0).cpu().numpy()

    return points


def main():
    parser = argparse.ArgumentParser(
        description="Compress a 3D point cloud using Squeeze3D with LION"
    )
    parser.add_argument(
        "pc_path", type=str, help="Path to input point cloud file (PLY)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output_pc.ply",
        help="Path for output point cloud",
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
        "--pointnet_weights",
        type=str,
        default=None,
        help="Path to PointNet++ weights (overrides default)",
    )
    parser.add_argument(
        "--lion_config", type=str, default=None, help="Path to LION config file"
    )
    parser.add_argument(
        "--lion_weights", type=str, default=None, help="Path to LION model weights"
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

    original_size = calculate_pc_size(args.pc_path)
    points = load_point_cloud(args.pc_path)
    num_points = len(points)

    device = torch.device(args.device)

    classifier, label_map = get_model_and_labels(
        "pointnet2_cls_ssg", 40, normal_channel=False
    )

    if args.pointnet_weights:
        checkpoint_path = args.pointnet_weights
    else:
        checkpoint_path = os.path.join(
            args.weights_dir, "pointnet2_ssg_wo_normals", "best_model.pth"
        )

    if not os.path.exists(checkpoint_path):
        print(f"Error: PointNet++ weights not found at {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path)
    classifier.load_state_dict(checkpoint["model_state_dict"])
    classifier.eval()
    classifier = classifier.to(device)

    pc_latent, original_num_points = encode_with_pointnet(
        args.pc_path, classifier, device
    )
    pc_size = pc_latent.numel() * 4

    pc_latent_cpu = pc_latent.cpu()
    del classifier
    del pc_latent
    torch.cuda.empty_cache()
    gc.collect()

    compression_model = LTpcOrtho(
        hidden_size=8192,
        dropout_prob=0.3,
    )

    if args.squeeze3d_weights:
        model_path = args.squeeze3d_weights
    else:
        model_path = os.path.join(args.weights_dir, "pc_pn_lion_8192.safetensors")

    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}")
        return

    compression_model.load_state_dict(load_file(model_path))
    compression_model.to(device).eval()

    pc_latent_gpu = pc_latent_cpu.to(device)
    with torch.no_grad():
        compressed, b_matrix = compression_model(pc_latent_gpu.unsqueeze(0))

        b_matrix_squeezed = b_matrix.squeeze(0)
        compressed_size = b_matrix_squeezed.numel() * 4

    compressed_cpu = compressed.cpu()
    b_matrix_cpu = b_matrix.squeeze(0).cpu()
    del compression_model
    del pc_latent_gpu
    del compressed
    del b_matrix
    del b_matrix_squeezed
    torch.cuda.empty_cache()
    gc.collect()

    if args.lion_config:
        config_path = args.lion_config
    else:
        config_path = os.path.join(
            args.weights_dir, "lion_ckpt", "unconditional", "all55", "cfg.yml"
        )

    if not os.path.exists(config_path):
        print(f"Error: LION config not found at {config_path}")
        return

    config.merge_from_file(config_path)
    lion = LION(config)

    if args.lion_weights:
        model_path = args.lion_weights
    else:
        model_path = os.path.join(
            args.weights_dir,
            "lion_ckpt",
            "unconditional",
            "all55",
            "checkpoints",
            "epoch_10999_iters_2100999.pt",
        )

    if not os.path.exists(model_path):
        print(f"Error: LION weights not found at {model_path}")
        return

    lion.load_model(model_path)

    compressed_gpu = compressed_cpu.to(device)
    points = decode_with_lion(compressed_gpu.squeeze(0), lion, device)

    save_as_ply(points, args.output_path)

    output_size = calculate_pc_size(args.output_path)

    compressed_path = args.output_path.replace(".ply", "_compressed.pt")
    torch.save(b_matrix_cpu, compressed_path)

    print(f"\nCompression ratio: {original_size / compressed_size:.2f}x")
    print(f"Compressed representation saved to: {compressed_path}")
    print(f"Reconstructed point cloud saved to: {args.output_path}")


if __name__ == "__main__":
    main()
