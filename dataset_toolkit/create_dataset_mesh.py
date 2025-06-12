import argparse
import io
import os
import random
import shutil
import tempfile
import gc

import numpy as np
import open3d as o3d
import torch
import trimesh
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from datasets import Dataset, concatenate_datasets
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from torchvision.transforms import v2
from tqdm import tqdm

from squeeze3d.utils.shape_utils import (
    decode,
    load_shape_pipeline,
    make_latents,
)
from squeeze3d.utils.ma_utils import (
    load_meshanything,
    process_mesh,
    process_mesh_to_pc,
)
from squeeze3d.utils.ma_utils import get_args as get_meshanything_args
from squeeze3d.utils.im_utils import (
    encode_instantmesh,
    load_instantmesh,
    get_render_cameras,
    load_zero123plus,
    removebg,
    render_frames,
    decode_im,
)
from squeeze3d.instantmesh.utils.camera_util import get_zero123plus_input_cameras
from squeeze3d.instantmesh.utils.mesh_util import save_obj_with_mtl
from squeeze3d.utils.lrm_utils import load_lrm, lrm_decode, lrm_encode
from squeeze3d.utils.utils import seed_all

# Set random seed for reproducibility
seed_all(3407)


def generate_similar_latents(batch_size, latent_dim, device, shape_pipeline, prompt):
    """Generate similar latents with small noise variations."""
    base_latent = make_latents(shape_pipeline, prompt=[prompt])
    latents = base_latent.repeat(batch_size, 1)
    noise = torch.randn_like(latents) * 0.01
    latents += noise
    return latents


def serialize_image(pil_img):
    """Convert PIL Image to bytes for storage."""
    img_byte_arr = io.BytesIO()
    pil_img.save(img_byte_arr, format="PNG")
    return img_byte_arr.getvalue()


def process_instantmesh(image_path, model, pipeline, config, device="cuda"):
    """Process a single image through InstantMesh model."""
    input_image = Image.open(image_path)
    input_image = removebg(input_image)

    # Generate novel views using Zero123Plus
    output_image = pipeline(input_image, num_inference_steps=75).images[0]

    # Process images for InstantMesh
    images = np.asarray(output_image, dtype=np.float32) / 255.0
    images = torch.from_numpy(images).permute(2, 0, 1).contiguous().float()
    images = rearrange(images, "c (n h) (m w) -> (n m) c h w", n=3, m=2)
    images = images.unsqueeze(0).to(device)
    images = v2.functional.resize(images, 320, interpolation=3, antialias=True).clamp(
        0, 1
    )

    # Generate latents
    with torch.no_grad():
        input_cameras = get_zero123plus_input_cameras(
            batch_size=1, radius=4.0 * 1.0
        ).to(device)
        latents = encode_instantmesh(model, images, input_cameras)

    return latents


def decode_instantmesh(model, latents, config, temp_dir):
    """Decode InstantMesh latents to mesh with careful memory management."""
    torch.cuda.empty_cache()

    with torch.no_grad():
        # Decode to planes
        planes = decode_im(model, latents)

        # Move planes to CPU to free up GPU memory
        planes_cpu = planes.cpu()
        del planes
        torch.cuda.empty_cache()

        # Move back to GPU for mesh extraction
        planes = planes_cpu.cuda()
        del planes_cpu
        torch.cuda.empty_cache()

        # Extract mesh with reduced texture resolution if needed
        mesh_out = model.extract_mesh(
            planes,
            use_texture_map=True,
            **config.infer_config,
        )
        vertices, faces, uvs, mesh_tex_idx, tex_map = mesh_out

        # Immediately move to CPU and clear GPU memory
        vertices_cpu = vertices.cpu()
        faces_cpu = faces.cpu()
        uvs_cpu = uvs.cpu()
        mesh_tex_idx_cpu = mesh_tex_idx.cpu()
        tex_map_cpu = tex_map.cpu()

        planes_cpu = planes.cpu()
        del vertices, faces, uvs, mesh_tex_idx, tex_map, planes, mesh_out
        torch.cuda.empty_cache()

        # Save to obj file
        mesh_path = os.path.join(temp_dir, "mesh.obj")
        save_obj_with_mtl(
            vertices_cpu.numpy(),
            uvs_cpu.numpy(),
            faces_cpu.numpy(),
            mesh_tex_idx_cpu.numpy(),
            tex_map_cpu.permute(1, 2, 0).numpy(),
            mesh_path,
        )

        return mesh_path, planes_cpu


def process_lrm(image_path, lrm_model, temp_dir):
    """Process a single image through LRM model."""
    if "preprocessor" not in globals():
        from squeeze3d.OpenLRM.openlrm.utils.preprocess import Preprocessor

        global preprocessor
        preprocessor = Preprocessor()
    with torch.no_grad():
        out = os.path.join(temp_dir, "preprocess.png")
        _ = preprocessor.preprocess(
            image_path=image_path, save_path=out, rmbg=True, recenter=True
        )
        lrm_latents = lrm_encode(out, lrm_model)
        _, mesh_path = lrm_decode(lrm_latents, lrm_model, False, True, temp_dir)
    return lrm_latents, mesh_path


def process_mesh_for_ma(
    method, mesh_path, meshanything, device, use_mesh_to_pc=False, ma_decode=False
):
    """Process mesh through MeshAnything."""
    if ma_decode:
        mesh = trimesh.load(mesh_path)
        pc_normal = process_mesh_to_pc([mesh])[0][0]
        pc_coor = pc_normal[:, :3]
        normals = pc_normal[:, 3:]
        bounds = np.array([pc_coor.min(axis=0), pc_coor.max(axis=0)])
        pc_coor = pc_coor - (bounds[0] + bounds[1])[None, :] / 2
        pc_coor = pc_coor / np.abs(pc_coor).max() * 0.9995
        assert (
            np.linalg.norm(normals, axis=-1) > 0.99
        ).all(), "normals should be unit vectors"
        inp = np.concatenate([pc_coor, normals], axis=-1, dtype=np.float16)
        outs = meshanything(
            torch.tensor(inp, device=device).unsqueeze(0), sampling=False
        )
        encoded_latent = outs[0].squeeze(0)
    else:
        if use_mesh_to_pc:
            if method == "lrm" or method == "shape":
                # mesh_o3d = o3d.io.read_triangle_mesh(mesh_path)
                # mesh_o3d.remove_duplicated_triangles()
                # mesh_o3d.remove_degenerate_triangles()
                # mesh_o3d.remove_duplicated_vertices()
                # mesh_o3d.remove_non_manifold_edges()
                # mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh_o3d)
                # repaired_mesh_t = mesh_t.fill_holes(hole_size=1000000.0)
                # repaired_mesh_o3d = repaired_mesh_t.to_legacy()
                # repaired_mesh = trimesh.Trimesh(
                #     vertices=np.asarray(repaired_mesh_o3d.vertices),
                #     faces=np.asarray(repaired_mesh_o3d.triangles),
                #     process=False,
                # )
                # repaired_mesh.fill_holes()
                # repaired_mesh = repaired_mesh.subdivide()
                # repaired_mesh = repaired_mesh.smoothed()
                repaired_mesh = trimesh.load(mesh_path)
                pc_normal = process_mesh_to_pc([repaired_mesh])[0][0]
            elif method == "instantmesh":
                pc_normal = process_mesh_to_pc([trimesh.load(mesh_path)])[0][0]
            pc_normal = torch.tensor(pc_normal, dtype=torch.float32).to(device)
        else:
            pc_normal = process_mesh(mesh_path)
            pc_normal = (
                torch.tensor(pc_normal, dtype=torch.float32).unsqueeze(0).to(device)
            )

        with torch.no_grad():
            point_feature = meshanything.point_encoder.encode_latents(
                pc_normal.unsqueeze(0) if use_mesh_to_pc else pc_normal
            )
            encoded_latent = meshanything.process_point_feature(point_feature)
            encoded_latent = encoded_latent.squeeze(0)

    return encoded_latent


def create_dataset(
    num_samples,
    batch_size,
    latent_dim,
    device,
    output_file,
    upload_to_hub,
    repo_id,
    token,
    fp16,
    no_noise,
    prompt_file,
    use_mesh_to_pc,
    model,
    lrm_model_name,
    lrm_config_path,
    image_dir,
    ma_decode,
    instantmesh_config=None,
):
    """Create dataset from different model types."""

    # Initialize appropriate model based on type
    if model == "shape":
        shape_pipeline = load_shape_pipeline(device, fp16=fp16)
    elif model == "lrm":
        lrm_model = load_lrm(lrm_model_name, lrm_config_path)
    elif model == "instantmesh":
        config = OmegaConf.load(instantmesh_config)
        instantmesh_model = load_instantmesh(config=instantmesh_config)
        zero123plus = load_zero123plus(config.infer_config, device)

    # Initialize MeshAnything
    meshanything_args = get_meshanything_args()
    meshanything = load_meshanything(meshanything_args, device=device)

    if ma_decode:
        AcceleratorState._reset_state()
        accelerator = Accelerator(mixed_precision="fp16", project_dir="./")
        meshanything = accelerator.prepare(meshanything)

    # Define chunk size for periodic saving
    CHUNK_SIZE = 500
    chunk_datasets = []
    
    inputs = []
    outputs = []
    renders = []
    planes_cpu = []

    def save_chunk():
        """Save current chunk and clear memory."""
        nonlocal inputs, outputs, renders, planes_cpu, chunk_datasets
        
        if not inputs:
            return
            
        if model == "shape" and renders:
            chunk_data = {
                "inputs": inputs,
                "outputs": [serialize_image(img) for img in outputs],
                "renders": renders,
            }
        elif model == "instantmesh":
            chunk_data = {"inputs": inputs, "outputs": outputs, "planes": planes_cpu}
        else:
            chunk_data = {"inputs": inputs, "outputs": outputs}
        
        chunk_dataset = Dataset.from_dict(chunk_data)
        chunk_datasets.append(chunk_dataset)
        
        # Clear lists to free memory
        inputs = []
        outputs = []
        renders = []
        planes_cpu = []
        
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache()

    if model == "instantmesh":
        images = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'))])
        pbar = tqdm(total=num_samples, desc="Processing InstantMesh samples")

        for i in range(min(num_samples, len(images))):
            image_path = os.path.join(image_dir, images[i])

            # Step 1: Generate latents using InstantMesh
            im_latents = process_instantmesh(
                image_path, instantmesh_model, zero123plus, config, device
            )

            # Clear Zero123Plus and other cached memory from GPU
            del zero123plus
            torch.cuda.empty_cache()

            # Move latents to CPU temporarily to free more GPU memory
            im_latents_cpu = im_latents.cpu()
            del im_latents
            torch.cuda.empty_cache()

            # Step 2: Decode latents to mesh and process through MeshAnything
            with tempfile.TemporaryDirectory() as temp_dir:
                # Move latents back to GPU for processing
                im_latents = im_latents_cpu.cuda()
                del im_latents_cpu
                torch.cuda.empty_cache()

                mesh_path, plane_cpu = decode_instantmesh(
                    instantmesh_model, im_latents, config, temp_dir
                )
                encoded_latent = process_mesh_for_ma(
                    model, mesh_path, meshanything, device, use_mesh_to_pc, ma_decode
                )

            inputs.append(encoded_latent.cpu().numpy().astype(np.float32))
            outputs.append(im_latents.squeeze(0).cpu().numpy().astype(np.float32))
            planes_cpu.append(plane_cpu.squeeze(0).numpy().astype(np.float32))
            pbar.update(1)

            # Save chunk periodically
            if (i + 1) % CHUNK_SIZE == 0:
                print(f"\nSaving chunk at sample {i + 1}...")
                save_chunk()

            # Reload Zero123Plus for next iteration
            if i < num_samples - 1:
                zero123plus = load_zero123plus(config.infer_config, device)

        pbar.close()

    elif model == "lrm":
        images = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'))])
        pbar = tqdm(total=num_samples, desc="Processing LRM samples")
        i = 0

        while i < num_samples:
            image_path = os.path.join(image_dir, images[i])

            with tempfile.TemporaryDirectory() as temp_dir:
                # Process through LRM
                lrm_latents, mesh_path = process_lrm(image_path, lrm_model, temp_dir)

                # Process through MeshAnything
                encoded_latent = process_mesh_for_ma(
                    model, mesh_path, meshanything, device, use_mesh_to_pc, ma_decode
                )

                inputs.append(encoded_latent.cpu().numpy().astype(np.float32))
                outputs.append(lrm_latents.squeeze(0).cpu().numpy().astype(np.float32))

            i += 1
            pbar.update(1)
            
            # Save chunk periodically
            if i % CHUNK_SIZE == 0:
                print(f"\nSaving chunk at sample {i}...")
                save_chunk()

        pbar.close()

    elif model == "shape":
        if prompt_file:
            with open(prompt_file, "r") as f:
                prompts = f.readlines()

        sample_count = 0
        for i in tqdm(
            range(0, num_samples, batch_size), desc="Processing Shape samples"
        ):
            prompt = (
                str(prompts[i % len(prompts)]) if prompt_file else "A birthday cupcake"
            )

            # Generate latents
            if no_noise:
                latents = make_latents(shape_pipeline, prompt=[prompt])
                latents = latents.repeat(batch_size, 1)
            else:
                latents = generate_similar_latents(
                    batch_size, latent_dim, device, shape_pipeline, prompt
                )

            with tempfile.TemporaryDirectory() as temp_dir:
                for j in range(batch_size):
                    # Decode to mesh
                    mesh = decode(
                        shape_pipeline,
                        latents[j].unsqueeze(0),
                        prompt=[""],
                        export_to=[os.path.join(temp_dir, f"mesh_{j}")],
                        return_trimesh=True,
                    )

                    # Process through MeshAnything
                    encoded_latent = process_mesh_for_ma(
                        model,
                        os.path.join(temp_dir, f"mesh_{j}.obj"),
                        meshanything,
                        device,
                        use_mesh_to_pc,
                        ma_decode,
                    )

                    if not use_mesh_to_pc:
                        images = decode(
                            shape_pipeline,
                            latents[j].unsqueeze(0),
                            [""],
                            False,
                            False,
                            [],
                            False,
                        )
                        renders.append(images.images[0][0])

                    inputs.append(encoded_latent.cpu().numpy().astype(np.float64))
                    outputs.append(latents[j].cpu().numpy().astype(np.float64))
                    
                    sample_count += 1
                    
                    # Save chunk periodically
                    if sample_count % CHUNK_SIZE == 0:
                        print(f"\nSaving chunk at sample {sample_count}...")
                        save_chunk()

    # Save any remaining samples
    if inputs:
        print(f"\nSaving final chunk with {len(inputs)} samples...")
        save_chunk()

    # Concatenate all chunks
    print(f"\nConcatenating {len(chunk_datasets)} chunks...")
    dataset = concatenate_datasets(chunk_datasets)

    # Save or upload dataset
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

    # General arguments
    parser.add_argument(
        "--num_samples", type=int, default=1000, help="Number of samples to generate"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for processing"
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=1024 * 1024,
        help="Dimension of the latent space",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="latent_dataset",
        help="Output file for the dataset",
    )

    # Dataset storage options
    parser.add_argument(
        "--upload_to_hub",
        action="store_true",
        help="Upload dataset to Hugging Face Hub",
    )
    parser.add_argument(
        "--repo_id", type=str, help="Repository ID for Hugging Face Hub upload"
    )
    parser.add_argument("--token", type=str, help="Hugging Face API token")

    # Processing options
    parser.add_argument("--fp16", action="store_true", help="Use FP16 for processing")
    parser.add_argument(
        "--no-noise", action="store_true", help="Disable noise addition to latents"
    )
    parser.add_argument(
        "--use_mesh_to_pc",
        action="store_true",
        help="Use mesh to point cloud conversion",
    )
    parser.add_argument(
        "--ma_decode", action="store_true", help="Use MeshAnything decoder"
    )

    # Model specific arguments
    parser.add_argument(
        "--model",
        type=str,
        default="shape",
        choices=["shape", "lrm", "instantmesh"],
        help="Model type to use (shape, lrm, or instantmesh)",
    )

    # Shape model specific arguments
    parser.add_argument(
        "--prompt_file", type=str, help="File containing prompts for shape model"
    )

    # LRM model specific arguments
    parser.add_argument("--lrm_model_name", type=str, help="Model name for LRM")
    parser.add_argument("--lrm_config_path", type=str, help="Config path for LRM")

    # InstantMesh specific arguments
    parser.add_argument(
        "--instantmesh_config", type=str, help="Config path for InstantMesh model"
    )

    # Common arguments for LRM and InstantMesh
    parser.add_argument(
        "--image_dir", type=str, help="Directory containing input images"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.upload_to_hub and (not args.repo_id or not args.token):
        parser.error("--repo_id and --token are required when --upload_to_hub is set")

    if args.model == "lrm" and (not args.lrm_model_name or not args.lrm_config_path):
        parser.error(
            "--lrm_model_name and --lrm_config_path are required for LRM model"
        )

    if args.model == "instantmesh" and not args.instantmesh_config:
        parser.error("--instantmesh_config is required for InstantMesh model")

    if args.model in ["lrm", "instantmesh"] and not args.image_dir:
        parser.error("--image_dir is required for LRM and InstantMesh models")

    create_dataset(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        device=args.device,
        output_file=args.output_file,
        upload_to_hub=args.upload_to_hub,
        repo_id=args.repo_id,
        token=args.token,
        fp16=args.fp16,
        no_noise=args.no_noise,
        prompt_file=args.prompt_file,
        use_mesh_to_pc=args.use_mesh_to_pc,
        model=args.model,
        lrm_model_name=args.lrm_model_name,
        lrm_config_path=args.lrm_config_path,
        image_dir=args.image_dir,
        ma_decode=args.ma_decode,
        instantmesh_config=args.instantmesh_config,
    )


if __name__ == "__main__":
    main()
