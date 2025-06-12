import torch
import numpy as np
import argparse
import os
import glob
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
    TexturesVertex,
)
import matplotlib.pyplot as plt
from tqdm import tqdm
from pytorch3d.transforms import euler_angles_to_matrix
import math
from torchmetrics.image import (
    MultiScaleStructuralSimilarityIndexMeasure,
    LearnedPerceptualImagePatchSimilarity,
)
from torchmetrics.functional.image.psnr import peak_signal_noise_ratio
import trimesh


def fibonacci_sphere(samples=100):
    """Generate evenly distributed points on a sphere using Fibonacci method."""
    points = []
    phi = math.pi * (3.0 - math.sqrt(5.0))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        # Convert to Euler angles
        r = math.sqrt(x * x + y * y + z * z)
        phi_angle = math.acos(y / r)
        theta_angle = math.atan2(z, x)

        x_angle = 0  # Simplify by keeping X rotation at 0
        y_angle = theta_angle  # Rotation around Y axis (azimuth)
        z_angle = phi_angle  # Rotation around Z after Y rotation

        points.append((x_angle, y_angle, z_angle))

    return points


def render_mesh_views(
    obj_path,
    texture_path,
    output_dir,
    n_views=36,
    image_size=512,
    fallback_verts_uvs=None,
    fallback_faces_uvs=None,
    ignore_texture=False,
):
    """
    Render n views of a mesh by rotating it in multiple directions.

    Args:
        obj_path: Path to the obj file
        texture_path: Path to the texture image
        output_dir: Directory to save rendered views
        n_views: Number of views to render
        image_size: Size of the rendered images
        fallback_verts_uvs: Optional fallback UV coordinates to use if obj file doesn't have them
        fallback_faces_uvs: Optional fallback UV faces to use if obj file doesn't have them
        ignore_texture: If True, use a solid gray color instead of the texture
    """
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda")

    try:
        verts, faces, aux = load_obj(obj_path)
    except Exception as e:
        print(f"Error loading mesh from {obj_path}: {e}")
        raise

    verts_orig = verts.to(device)
    faces_idx = faces.verts_idx.to(device)

    # Handle textures based on ignore_texture flag
    if ignore_texture:
        # Create uniform gray vertex texture
        verts_rgb = torch.ones_like(verts_orig)[None]  # (1, V, 3)
        verts_rgb = verts_rgb * 0.5  # Gray color
        textures = TexturesVertex(verts_features=verts_rgb)
    else:
        # Use normal texture mapping
        # Check if UV coordinates exist in the loaded mesh
        if aux.verts_uvs is not None and faces.textures_idx is not None:
            print(f"Using UV coordinates from {obj_path}")
            verts_uvs = aux.verts_uvs[None, ...].to(device)  # (1, V, 2)
            faces_uvs = faces.textures_idx[None, ...].to(device)  # (1, F, 3)
        elif fallback_verts_uvs is not None and fallback_faces_uvs is not None:
            print(f"Using fallback UV coordinates for {obj_path}")
            verts_uvs = fallback_verts_uvs
            faces_uvs = fallback_faces_uvs
        else:
            raise ValueError(
                f"No UV coordinates found in {obj_path} and no fallback provided"
            )

        texture_image = plt.imread(texture_path)[..., :3]
        texture_image = torch.from_numpy(texture_image).to(device).unsqueeze(0)

        textures = TexturesUV(
            maps=texture_image, faces_uvs=faces_uvs, verts_uvs=verts_uvs
        )

    R, T = look_at_view_transform(2.5, elev=10, azim=0)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    raster_settings = RasterizationSettings(
        image_size=image_size, blur_radius=0.0, faces_per_pixel=1
    )

    lights = PointLights(device=device, location=[[0.0, 0.0, 3.0]])

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights),
    )

    rotation_angles = fibonacci_sphere(n_views)

    rendered_views = []

    for i, (x_angle, y_angle, z_angle) in enumerate(rotation_angles):
        rot_angles = torch.tensor([x_angle, y_angle, z_angle], device=device).unsqueeze(
            0
        )
        rot_matrix = euler_angles_to_matrix(rot_angles, "XYZ")

        rotated_verts = torch.matmul(verts_orig, rot_matrix.squeeze(0).transpose(0, 1))

        mesh = Meshes(verts=[rotated_verts], faces=[faces_idx], textures=textures)

        images = renderer(mesh)

        # Ensure consistent RGB format without alpha
        rendered_image = images[0, ..., :3]

        # Ensure values are in expected range [0, 1]
        rendered_image = rendered_image.clamp(0, 1)

        rendered_views.append(rendered_image)

        # Save the image if output_dir is provided
        # if output_dir:
        #     plt.figure(figsize=(10, 10))
        #     plt.imshow(rendered_image.cpu().numpy())
        #     plt.axis("off")
        #     plt.tight_layout()
        #     plt.savefig(os.path.join(output_dir, f"view_{i:03d}.png"),
        #                bbox_inches='tight', pad_inches=0)
        #     plt.close()

    return rendered_views


def calculate_metrics(gt_views, gen_views):
    """Calculate image quality metrics between ground truth and generated views."""
    psnr_values = []
    ms_ssim_values = []
    lpips_values = []

    ms_ssim_fn = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to("cuda")
    lpips_fn = LearnedPerceptualImagePatchSimilarity().to("cuda")

    for i, (gt_img, gen_img) in enumerate(zip(gt_views, gen_views)):
        # Ensure we're only using RGB channels (no alpha)
        if gt_img.shape[-1] > 3:
            gt_img = gt_img[..., :3]
        if gen_img.shape[-1] > 3:
            gen_img = gen_img[..., :3]

        # Ensure both images are on the same device
        if gt_img.device != gen_img.device:
            gen_img = gen_img.to(gt_img.device)

        # Check for NaN or inf values
        if (
            torch.isnan(gt_img).any()
            or torch.isnan(gen_img).any()
            or torch.isinf(gt_img).any()
            or torch.isinf(gen_img).any()
        ):
            print(
                f"Warning: NaN or inf values detected in view pair {i}. Skipping this pair."
            )
            continue

        # Ensure values are in the expected range [0, 1]
        gt_img = gt_img.clamp(0, 1)
        gen_img = gen_img.clamp(0, 1)

        # For PSNR
        try:
            psnr = peak_signal_noise_ratio(gen_img, gt_img)
            if not torch.isnan(psnr) and not torch.isinf(psnr):
                psnr_values.append(psnr.item())
            else:
                print(f"Warning: PSNR calculation resulted in NaN or inf for view {i}")
        except Exception as e:
            print(f"Error calculating PSNR for view {i}: {e}")

        # For MS-SSIM
        try:
            # Convert from (H, W, C) to (1, C, H, W)
            gt_ssim = gt_img.permute(2, 0, 1).unsqueeze(0)
            gen_ssim = gen_img.permute(2, 0, 1).unsqueeze(0)

            ms_ssim = ms_ssim_fn(gen_ssim, gt_ssim)
            if not torch.isnan(ms_ssim) and not torch.isinf(ms_ssim):
                ms_ssim_values.append(ms_ssim.item())
            else:
                print(
                    f"Warning: MS-SSIM calculation resulted in NaN or inf for view {i}"
                )
        except Exception as e:
            print(f"Error calculating MS-SSIM for view {i}: {e}")

        # For LPIPS
        try:
            # Normalize to [-1, 1] range
            gt_lpips = gt_img.permute(2, 0, 1).unsqueeze(0).clamp(0, 1) * 2 - 1
            gen_lpips = gen_img.permute(2, 0, 1).unsqueeze(0).clamp(0, 1) * 2 - 1

            # Verify no NaN or inf values in normalized tensors
            if (
                not torch.isnan(gt_lpips).any()
                and not torch.isnan(gen_lpips).any()
                and not torch.isinf(gt_lpips).any()
                and not torch.isinf(gen_lpips).any()
            ):
                lpips = lpips_fn(gen_lpips, gt_lpips)
                if not torch.isnan(lpips) and not torch.isinf(lpips):
                    lpips_values.append(lpips.item())
                else:
                    print(
                        f"Warning: LPIPS calculation resulted in NaN or inf for view {i}"
                    )
            else:
                print(
                    f"Warning: NaN or inf values detected after normalization for LPIPS for view {i}"
                )
        except Exception as e:
            print(f"Error calculating LPIPS for view {i}: {e}")

    # Return metrics if we have valid calculations, otherwise use None
    return {
        "psnr": np.mean(psnr_values) if psnr_values else None,
        "ms_ssim": np.mean(ms_ssim_values) if ms_ssim_values else None,
        "lpips": np.mean(lpips_values) if lpips_values else None,
    }


def process_directory(
    gt_dir,
    gen_dir,
    subdirectory,
    output_base_dir,
    n_views=36,
    image_size=512,
    type="draco",
    ignore_texture=False,
):
    """Process a specific subdirectory to compare ground truth and generated meshes."""
    try:
        with torch.no_grad():
            if type == "draco":
                gt_mesh_path = os.path.join(gt_dir, subdirectory, "mesh.obj")
                gt_texture_path = os.path.join(gt_dir, subdirectory, "mesh.png")

                gen_mesh_path = os.path.join(gen_dir, subdirectory, "decomp.obj")
                gen_texture_path = os.path.join(gt_dir, subdirectory, "mesh.png")
            elif type == "instantmesh":
                gt_mesh_path = os.path.join(gt_dir, subdirectory, "mesh.obj")
                gt_texture_path = os.path.join(gt_dir, subdirectory, "mesh.png")

                gen_mesh_path = os.path.join(gen_dir, subdirectory, "mesh.obj")
                gen_texture_path = os.path.join(gen_dir, subdirectory, "mesh.png")
            elif type == "ngf":
                gt_mesh_path = os.path.join(gt_dir, subdirectory, "mesh.obj")
                gt_texture_path = os.path.join(gt_dir, subdirectory, "mesh.png")

                gen_mesh_path = os.path.join(
                    gen_dir, subdirectory, subdirectory, "decomp.stl"
                )
                m = trimesh.load(gen_mesh_path)
                gen_mesh_path = os.path.join(
                    gen_dir, subdirectory, subdirectory, "mesh.obj"
                )
                m.export(gen_mesh_path)
                gen_texture_path = os.path.join(gt_dir, subdirectory, "mesh.png")

            gt_output_dir = os.path.join(output_base_dir, subdirectory, "gt_views")
            gen_output_dir = os.path.join(output_base_dir, subdirectory, "gen_views")

            if not os.path.exists(gt_mesh_path):
                print(f"Warning: Ground truth mesh not found at {gt_mesh_path}")
                return None

            if not ignore_texture and not os.path.exists(gt_texture_path):
                print(f"Warning: Ground truth texture not found at {gt_texture_path}")
                return None

            if not os.path.exists(gen_mesh_path):
                print(f"Warning: Generated mesh not found at {gen_mesh_path}")
                return None

            # First load ground truth mesh to extract UV coordinates for possible fallback
            device = torch.device("cuda")

            try:
                _, gt_faces, gt_aux = load_obj(gt_mesh_path)

                # Extract ground truth UV coordinates as fallback
                gt_verts_uvs = None
                gt_faces_uvs = None

                if not ignore_texture:
                    if (
                        gt_aux.verts_uvs is not None
                        and gt_faces.textures_idx is not None
                    ):
                        gt_verts_uvs = gt_aux.verts_uvs[None, ...].to(device)
                        gt_faces_uvs = gt_faces.textures_idx[None, ...].to(device)
                    else:
                        print(
                            f"Warning: Ground truth mesh {gt_mesh_path} doesn't have UV coordinates"
                        )
                        if not ignore_texture:
                            return None

                print(f"Rendering ground truth views for {subdirectory}...")
                gt_views = render_mesh_views(
                    gt_mesh_path,
                    gt_texture_path,
                    gt_output_dir,
                    n_views,
                    image_size,
                    fallback_verts_uvs=gt_verts_uvs,
                    fallback_faces_uvs=gt_faces_uvs,
                    ignore_texture=ignore_texture,
                )

                print(f"Rendering generated views for {subdirectory}...")
                # Pass ground truth UV coordinates as fallback for generated mesh
                gen_views = render_mesh_views(
                    gen_mesh_path,
                    gen_texture_path,
                    gen_output_dir,
                    n_views,
                    image_size,
                    fallback_verts_uvs=gt_verts_uvs,
                    fallback_faces_uvs=gt_faces_uvs,
                    ignore_texture=ignore_texture,
                )

                print(f"Calculating metrics for {subdirectory}...")
                metrics = calculate_metrics(gt_views, gen_views)

                return metrics
            except Exception as e:
                print(f"Error processing {subdirectory}: {e}")
                return None
    except Exception as e:
        print(f"Unexpected error processing {subdirectory}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Compare rendered views of ground truth and generated meshes"
    )
    parser.add_argument(
        "--gt_dir", required=True, help="Directory containing ground truth meshes"
    )
    parser.add_argument(
        "--gen_dir", required=True, help="Directory containing generated meshes"
    )
    parser.add_argument(
        "--output_dir", default="comparison_results", help="Directory to save outputs"
    )
    parser.add_argument(
        "--type",
        default="draco",
        choices=["draco", "instantmesh", "ngf"],
        help="Type of compression",
    )
    parser.add_argument(
        "--n_views", type=int, default=36, help="Number of views to render per mesh"
    )
    parser.add_argument(
        "--image_size", type=int, default=512, help="Size of rendered images"
    )
    parser.add_argument(
        "--ignore_texture",
        action="store_true",
        help="Ignore texture and use solid gray color for geometry comparison only",
    )

    args = parser.parse_args()

    if args.ignore_texture:
        print(
            "Texture will be ignored. Using solid gray color for geometry comparison only."
        )

    subdirectories = []
    for item in os.listdir(args.gt_dir):
        if os.path.isdir(os.path.join(args.gt_dir, item)):
            subdirectories.append(item)

    if not subdirectories:
        print(f"No subdirectories found in {args.gt_dir}")
        return

    all_metrics = {}

    for subdir in subdirectories:
        print(f"\nProcessing {subdir}...")
        metrics = process_directory(
            args.gt_dir,
            args.gen_dir,
            subdir,
            args.output_dir,
            args.n_views,
            args.image_size,
            args.type,
            args.ignore_texture,
        )

        if metrics:
            all_metrics[subdir] = metrics
            print(f"Metrics for {subdir}:")
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

    if all_metrics:
        # Filter out None values for each metric
        valid_psnr = [m["psnr"] for m in all_metrics.values() if m["psnr"] is not None]
        valid_ms_ssim = [
            m["ms_ssim"] for m in all_metrics.values() if m["ms_ssim"] is not None
        ]
        valid_lpips = [
            m["lpips"] for m in all_metrics.values() if m["lpips"] is not None
        ]

        # Calculate averages if there are valid metrics
        avg_psnr = np.mean(valid_psnr) if valid_psnr else None
        avg_ms_ssim = np.mean(valid_ms_ssim) if valid_ms_ssim else None
        avg_lpips = np.mean(valid_lpips) if valid_lpips else None

        # Calculate standard deviations
        std_psnr = np.std(valid_psnr) if valid_psnr else None
        std_ms_ssim = np.std(valid_ms_ssim) if valid_ms_ssim else None
        std_lpips = np.std(valid_lpips) if valid_lpips else None

        print("\nAverage metrics across all directories:")
        if avg_psnr is not None:
            print(f"  Average PSNR: {avg_psnr:.4f} (std: {std_psnr:.4f})")
        else:
            print(f"  Average PSNR: No valid calculations")

        if avg_ms_ssim is not None:
            print(f"  Average MS-SSIM: {avg_ms_ssim:.4f} (std: {std_ms_ssim:.4f})")
        else:
            print(f"  Average MS-SSIM: No valid calculations")

        if avg_lpips is not None:
            print(f"  Average LPIPS: {avg_lpips:.4f} (std: {std_lpips:.4f})")
        else:
            print(f"  Average LPIPS: No valid calculations")

        results_file = os.path.join(
            args.output_dir,
            f"metrics_summary{'_no_texture' if args.ignore_texture else ''}.txt",
        )
        os.makedirs(args.output_dir, exist_ok=True)

        with open(results_file, "w") as f:
            f.write("Mesh Comparison Results\n")
            f.write("======================\n\n")

            if args.ignore_texture:
                f.write(
                    "Evaluation performed with textures ignored (geometry only).\n\n"
                )

            # for subdir, metrics in all_metrics.items():
            #     f.write(f"Metrics for {subdir}:\n")
            #     f.write(f"  PSNR: {metrics['psnr']:.4f if metrics['psnr'] is not None else 'Failed'}\n")
            #     f.write(f"  MS-SSIM: {metrics['ms_ssim']:.4f if metrics['ms_ssim'] is not None else 'Failed'}\n")
            #     f.write(f"  LPIPS: {metrics['lpips']:.4f if metrics['lpips'] is not None else 'Failed'}\n\n")

            f.write("Average metrics across all directories:\n")
            if avg_psnr is not None:
                f.write(f"  Average PSNR: {avg_psnr:.4f} (std: {std_psnr:.4f})\n")
            else:
                f.write(f"  Average PSNR: No valid calculations\n")

            if avg_ms_ssim is not None:
                f.write(
                    f"  Average MS-SSIM: {avg_ms_ssim:.4f} (std: {std_ms_ssim:.4f})\n"
                )
            else:
                f.write(f"  Average MS-SSIM: No valid calculations\n")

            if avg_lpips is not None:
                f.write(f"  Average LPIPS: {avg_lpips:.4f} (std: {std_lpips:.4f})\n")
            else:
                f.write(f"  Average LPIPS: No valid calculations\n")

        print(f"Results saved to {results_file}")
    else:
        print("No valid results to report")


if __name__ == "__main__":
    main()
