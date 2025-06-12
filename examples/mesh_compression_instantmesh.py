import os
import sys
import torch
import trimesh
import numpy as np
import argparse
from pathlib import Path
import tempfile
from omegaconf import OmegaConf
import gc

from squeeze3d.utils.im_utils import load_instantmesh, decode_im
from squeeze3d.utils.ma_utils import load_meshanything, get_args, process_mesh_to_pc
from squeeze3d.instantmesh.utils.mesh_util import save_obj_with_mtl
from squeeze3d.models.mlp import LTOrtho
from safetensors.torch import load_file
from squeeze3d.utils.utils import seed_all


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from squeeze3d.metrics.mesh_metrics import render_mesh_views, calculate_metrics

seed_all(3407)


def calculate_mesh_size(mesh_path):
    total_size = 0
    mesh_path = Path(mesh_path)

    if mesh_path.exists():
        total_size += mesh_path.stat().st_size

    if mesh_path.suffix.lower() == ".obj":
        mtl_path = mesh_path.with_suffix(".mtl")
        if mtl_path.exists():
            total_size += mtl_path.stat().st_size

            try:
                with open(mtl_path, "r") as f:
                    for line in f:
                        line = line.strip()

                        if line.startswith(
                            (
                                "map_Kd",
                                "map_Ka",
                                "map_Ks",
                                "map_Ns",
                                "map_d",
                                "map_bump",
                                "bump",
                            )
                        ):
                            parts = line.split()
                            if len(parts) > 1:
                                texture_path = mesh_path.parent / parts[-1]
                                if texture_path.exists():
                                    total_size += texture_path.stat().st_size
            except:
                pass

    return total_size


def process_mesh_for_ma(mesh_path, meshanything, device):

    mesh = trimesh.load(mesh_path)
    pc_normal = process_mesh_to_pc([mesh])[0][0]
    pc_normal = torch.tensor(pc_normal, dtype=torch.float32).to(device)

    with torch.no_grad():
        point_feature = meshanything.point_encoder.encode_latents(
            pc_normal.unsqueeze(0)
        )
        encoded_latent = meshanything.process_point_feature(point_feature)
        encoded_latent = encoded_latent.squeeze(0)

    return encoded_latent


def decode_with_instantmesh(compressed_output, instantmesh_model, config):
    with torch.no_grad():

        mesh_out = instantmesh_model.extract_mesh(
            compressed_output,
            use_texture_map=True,
            **config.infer_config,
        )
        vertices, faces, uvs, mesh_tex_idx, tex_map = mesh_out

    return vertices, faces, uvs, mesh_tex_idx, tex_map


def main():
    parser = argparse.ArgumentParser(
        description="Compress a 3D mesh using Squeeze3D with InstantMesh"
    )
    parser.add_argument("mesh_path", type=str, help="Path to input mesh file")
    parser.add_argument(
        "--output_path",
        type=str,
        default="output_mesh.obj",
        help="Path for output mesh",
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
        "--instantmesh_config",
        type=str,
        default="weights/instantmesh/instant-mesh-large.yaml",
        help="Path to InstantMesh config file",
    )

    parser.add_argument(
        "--squeeze3d_weights",
        type=str,
        default=None,
        help="Path to Squeeze3D model weights (overrides weights_dir)",
    )
    parser.add_argument(
        "--meshanything_weights",
        type=str,
        default=None,
        help="Path to MeshAnything model weights (overrides default)",
    )
    parser.add_argument(
        "--instantmesh_weights",
        type=str,
        default=None,
        help="Path to InstantMesh model weights (overrides default)",
    )

    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate reconstruction quality using LPIPS and other metrics",
    )
    parser.add_argument(
        "--n_views",
        type=int,
        default=36,
        help="Number of views to render for evaluation",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
        help="Size of rendered images for evaluation",
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

    mesh = trimesh.load(args.mesh_path)
    original_size = calculate_mesh_size(args.mesh_path)
    original_vertices = len(mesh.vertices)
    original_faces = len(mesh.faces)

    device = torch.device(args.device)
    config = OmegaConf.load(args.instantmesh_config)

    meshanything_args = get_args()
    if args.meshanything_weights:
        meshanything_args.model_path = args.meshanything_weights
    meshanything = load_meshanything(meshanything_args, device="cpu")
    meshanything = meshanything.to(device)

    ma_latent = process_mesh_for_ma(args.mesh_path, meshanything, device)
    ma_size = ma_latent.numel() * 4

    ma_latent_cpu = ma_latent.cpu()
    del meshanything
    del ma_latent
    torch.cuda.empty_cache()
    gc.collect()

    compression_model = LTOrtho(
        input_shape=(257, 1024),
        hidden_size=770,
        output_size=3 * 80 * 64 * 64,
        dropout_prob=0.35,
    )

    if args.squeeze3d_weights:
        model_path = args.squeeze3d_weights
    else:
        model_path = os.path.join(args.weights_dir, "mesh_ma_instantmesh.safetensors")

    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}")
        return

    compression_model.load_state_dict(load_file(model_path))
    compression_model.to(device).eval()

    ma_latent_gpu = ma_latent_cpu.to(device)
    with torch.no_grad():
        compressed, b_matrix = compression_model(ma_latent_gpu.unsqueeze(0))

        b_matrix_squeezed = b_matrix.squeeze(0)
        compressed_size = b_matrix_squeezed.numel() * 4

    compressed_cpu = compressed.cpu()
    b_matrix_cpu = b_matrix.squeeze(0).cpu()
    del compression_model
    del ma_latent_gpu
    del compressed
    del b_matrix
    del b_matrix_squeezed
    torch.cuda.empty_cache()
    gc.collect()

    if args.instantmesh_weights:
        config.model_config.init_from = args.instantmesh_weights

    instantmesh_model = load_instantmesh(config=args.instantmesh_config)

    compressed_gpu = compressed_cpu.to(device)
    compressed_reshaped = compressed_gpu.reshape(1, 3, 80, 64, 64)

    vertices, faces, uvs, mesh_tex_idx, tex_map = decode_with_instantmesh(
        compressed_reshaped, instantmesh_model, config
    )

    save_obj_with_mtl(
        vertices.cpu().numpy(),
        uvs.cpu().numpy(),
        faces.cpu().numpy(),
        mesh_tex_idx.cpu().numpy(),
        tex_map.permute(1, 2, 0).cpu().numpy(),
        args.output_path,
    )

    output_size = calculate_mesh_size(args.output_path)

    compressed_path = args.output_path.replace(".obj", "_compressed.pt")
    torch.save(b_matrix_cpu, compressed_path)

    print(f"\nCompression ratio: {original_size / compressed_size:.2f}x")
    print(f"Compressed representation saved to: {compressed_path}")
    print(f"Reconstructed mesh saved to: {args.output_path}")

    if args.evaluate:
        print("\nEvaluating reconstruction quality...")

        try:

            original_texture_path = None
            mesh_path = Path(args.mesh_path)

            possible_textures = [
                mesh_path.with_suffix(".png"),
                mesh_path.with_suffix(".jpg"),
                mesh_path.with_suffix(".jpeg"),
                mesh_path.parent / (mesh_path.stem + "_texture.png"),
                mesh_path.parent / "texture.png",
                mesh_path.parent / "mesh.png",
            ]

            for tex_path in possible_textures:
                if tex_path.exists():
                    original_texture_path = str(tex_path)
                    break

            reconstructed_texture_path = args.output_path.replace(".obj", ".png")

            has_original_texture = original_texture_path is not None and os.path.exists(
                original_texture_path
            )
            has_reconstructed_texture = os.path.exists(reconstructed_texture_path)

            with tempfile.TemporaryDirectory() as temp_dir:
                if has_original_texture and has_reconstructed_texture:
                    print("Rendering views for LPIPS evaluation with textures...")

                    original_views = render_mesh_views(
                        args.mesh_path,
                        original_texture_path,
                        os.path.join(temp_dir, "original_views"),
                        args.n_views,
                        args.image_size,
                        ignore_texture=False,
                    )

                    reconstructed_views = render_mesh_views(
                        args.output_path,
                        reconstructed_texture_path,
                        os.path.join(temp_dir, "reconstructed_views"),
                        args.n_views,
                        args.image_size,
                        ignore_texture=False,
                    )

                    print("Calculating image quality metrics...")
                    metrics = calculate_metrics(original_views, reconstructed_views)

                    print("\nReconstructed Mesh Quality Metrics (with texture):")
                    if metrics["psnr"] is not None:
                        print(f"  PSNR: {metrics['psnr']:.4f}")
                    else:
                        print(f"  PSNR: Failed to calculate")

                    if metrics["ms_ssim"] is not None:
                        print(f"  MS-SSIM: {metrics['ms_ssim']:.4f}")
                    else:
                        print(f"  MS-SSIM: Failed to calculate")

                    if metrics["lpips"] is not None:
                        print(f"  LPIPS: {metrics['lpips']:.4f}")
                    else:
                        print(f"  LPIPS: Failed to calculate")

                else:
                    if has_original_texture:
                        print(
                            "Reconstructed texture not found, evaluating geometry only..."
                        )
                    elif has_reconstructed_texture:
                        print("Original texture not found, evaluating geometry only...")
                    else:
                        print("No textures found, evaluating geometry only...")

                    original_views = render_mesh_views(
                        args.mesh_path,
                        "",
                        os.path.join(temp_dir, "original_views_geometry"),
                        args.n_views,
                        args.image_size,
                        ignore_texture=True,
                    )

                    reconstructed_views = render_mesh_views(
                        args.output_path,
                        "",
                        os.path.join(temp_dir, "reconstructed_views_geometry"),
                        args.n_views,
                        args.image_size,
                        ignore_texture=True,
                    )

                    print("Calculating geometry quality metrics...")
                    metrics = calculate_metrics(original_views, reconstructed_views)

                    print("\nReconstructed Mesh Geometry Quality Metrics:")
                    if metrics["psnr"] is not None:
                        print(f"  PSNR: {metrics['psnr']:.4f}")
                    else:
                        print(f"  PSNR: Failed to calculate")

                    if metrics["ms_ssim"] is not None:
                        print(f"  MS-SSIM: {metrics['ms_ssim']:.4f}")
                    else:
                        print(f"  MS-SSIM: Failed to calculate")

                    if metrics["lpips"] is not None:
                        print(f"  LPIPS: {metrics['lpips']:.4f}")
                    else:
                        print(f"  LPIPS: Failed to calculate")

        except Exception as e:
            print(f"Error during evaluation: {e}")
            import traceback

            traceback.print_exc()
            print("Continuing without evaluation...")


if __name__ == "__main__":
    main()
