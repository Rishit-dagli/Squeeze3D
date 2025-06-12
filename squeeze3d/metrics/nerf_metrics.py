#!/usr/bin/env python3
import os
import sys
import torch
import numpy as np
import argparse
from tqdm import tqdm
import torch.nn.functional as F
from torchmetrics.image import (
    MultiScaleStructuralSimilarityIndexMeasure,
    LearnedPerceptualImagePatchSimilarity,
)
from torchmetrics.functional.image.psnr import peak_signal_noise_ratio
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# Import functions from encode_decode.py
from squeeze3d.nerf.encode_decode import (
    load_data,
    density_to_alpha,
    save_npz_radiance_field,
)


def fibonacci_sphere(samples=20):
    """Generate evenly distributed points on a sphere using Fibonacci method."""
    points = []
    phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append((x, y, z))

    return points


def load_radiance_field(npz_path, device="cuda"):
    """
    Load a radiance field from an NPZ file, using the load_data function from encode_decode.py

    Args:
        npz_path: Path to the NPZ file
        device: Device to load the tensor on

    Returns:
        radiance_field: Radiance field tensor [1, 4, 160, 160, 160]
    """
    print(f"Loading radiance field from {npz_path}")
    try:
        # Use the load_data function from encode_decode.py
        radiance_field = load_data(npz_path, resolution=160, normalize_density=True)
        return radiance_field.to(device)

    except Exception as e:
        print(f"Error loading radiance field: {e}")
        # Try to fall back to more permissive loading approach
        try:
            # Load the NPZ file
            data = np.load(npz_path, allow_pickle=True)

            # Print available keys
            print(f"Available keys in NPZ file: {list(data.keys())}")

            # Check for specific NeRF-MAE keys
            if "rgbsigma" in data:
                # This is the standard NeRF-MAE format
                rgbsigma_np = data["rgbsigma"]
                print(f"Using 'rgbsigma' key with shape {rgbsigma_np.shape}")

                # Check if we need to transpose from (W, L, H, C) to (C, W, L, H)
                if len(rgbsigma_np.shape) == 4 and rgbsigma_np.shape[-1] <= 4:
                    # Format is [H, W, D, C] - need to transpose
                    print(f"Converting from [H, W, D, C] to [C, H, W, D]")
                    rgbsigma_np = np.transpose(rgbsigma_np, (3, 0, 1, 2))

                rgbsigma = torch.tensor(rgbsigma_np, dtype=torch.float32)

                # Normalize to [0, 1] if needed
                if rgbsigma.dtype == torch.uint8:
                    print("Converting uint8 values to float in range [0, 1]")
                    rgbsigma = rgbsigma.float() / 255.0

            elif "features" in data:
                features_np = data["features"]
                print(f"Using 'features' key with shape {features_np.shape}")

                # Load features in the format expected by NeRF-MAE
                if len(features_np.shape) == 4 and features_np.shape[-1] == 4:
                    # Format is [H, W, D, 4] with RGBA channels
                    print(f"Converting from [H, W, D, C] to [C, H, W, D]")
                    rgbsigma_np = np.transpose(features_np, (3, 0, 1, 2))
                    rgbsigma = torch.tensor(rgbsigma_np, dtype=torch.float32)
                else:
                    # General case - handle as regular tensor
                    rgbsigma = torch.tensor(features_np, dtype=torch.float32)
                    if len(rgbsigma.shape) == 4 and rgbsigma.shape[0] == 4:
                        # Already in [C, H, W, D] format
                        pass
                    elif len(rgbsigma.shape) == 4 and rgbsigma.shape[-1] == 4:
                        # Format is [H, W, D, C]
                        print(f"Converting from [H, W, D, C] to [C, H, W, D]")
                        rgbsigma = rgbsigma.permute(3, 0, 1, 2)

            elif "resolution" in data and any(
                k in data for k in ["rgbsigma", "grid", "latent"]
            ):
                # Handle more complex NeRF-MAE format with resolution and other metadata
                if "rgbsigma" in data:
                    rgbsigma_np = data["rgbsigma"]
                elif "grid" in data:
                    rgbsigma_np = data["grid"]
                elif "latent" in data:
                    rgbsigma_np = data["latent"]
                else:
                    # Use first available key as fallback
                    key = [k for k in data.keys() if k != "resolution"][0]
                    rgbsigma_np = data[key]
                    print(f"Using '{key}' key with shape {rgbsigma_np.shape}")

                # Check format and transpose if needed
                if len(rgbsigma_np.shape) == 4 and rgbsigma_np.shape[-1] <= 4:
                    # Assuming [H, W, D, C] format
                    print(f"Converting from [H, W, D, C] to [C, H, W, D]")
                    rgbsigma_np = np.transpose(rgbsigma_np, (3, 0, 1, 2))

                rgbsigma = torch.tensor(rgbsigma_np, dtype=torch.float32)

            else:
                # Handle feature vector format (many channels)
                key = list(data.keys())[0]  # Get first key
                data_np = data[key]
                print(f"Using first key '{key}' with shape {data_np.shape}")

                # Check if we have a feature vector format (many channels)
                if len(data_np.shape) == 4 and data_np.shape[-1] > 4:
                    print(
                        f"Detected feature vector format with {data_np.shape[-1]} channels"
                    )
                    # Format is [H, W, D, C] with many feature channels

                    # Extract first 3 channels for RGB
                    rgb_np = data_np[..., :3]

                    # For density (4th channel), use channel 3 if available
                    if data_np.shape[-1] > 3:
                        density_np = data_np[..., 3:4]

                        # Use density_to_alpha from encode_decode.py if possible
                        try:
                            alpha = density_to_alpha(density_np[..., 0])
                            density_np = alpha.reshape(*alpha.shape, 1)
                        except:
                            pass
                    else:
                        # Generate a simple sphere for density if not available
                        print("Generating placeholder density channel")
                        h, w, d = data_np.shape[:3]
                        center_x, center_y, center_z = h // 2, w // 2, d // 2
                        x, y, z = np.indices((h, w, d))

                        radius = min(h, w, d) * 0.4
                        sphere = (
                            (
                                (x - center_x) ** 2
                                + (y - center_y) ** 2
                                + (z - center_z) ** 2
                            )
                            < radius**2
                        ).astype(np.float32)
                        density_np = sphere.reshape(*sphere.shape, 1)

                    # Combine RGB and density to create rgbsigma
                    rgbsigma_np = np.concatenate([rgb_np, density_np], axis=-1)

                    # Convert from [H, W, D, C] to [C, H, W, D]
                    rgbsigma_np = np.transpose(rgbsigma_np, (3, 0, 1, 2))
                    rgbsigma = torch.tensor(rgbsigma_np, dtype=torch.float32)
                else:
                    # Regular format, convert to tensor
                    rgbsigma = torch.tensor(data_np, dtype=torch.float32)

                    # Get the format into [C, H, W, D] or [B, C, H, W, D]
                    if len(rgbsigma.shape) == 4:
                        if rgbsigma.shape[-1] <= 4:  # [H, W, D, C] format
                            print(f"Converting from [H, W, D, C] to [C, H, W, D]")
                            rgbsigma = rgbsigma.permute(3, 0, 1, 2)  # [C, H, W, D]
                    elif len(rgbsigma.shape) == 5:
                        if rgbsigma.shape[-1] <= 4:  # [B, H, W, D, C] format
                            print(f"Converting from [B, H, W, D, C] to [B, C, H, W, D]")
                            rgbsigma = rgbsigma.permute(
                                0, 4, 1, 2, 3
                            )  # [B, C, H, W, D]

            # From this point, ensure the tensor is in the right format
            print(f"Tensor shape after initial processing: {rgbsigma.shape}")

            # Ensure we have 4 channels (RGB + density)
            if len(rgbsigma.shape) == 4:  # [C, H, W, D] format
                if rgbsigma.shape[0] == 4:
                    # Already has 4 channels, perfect
                    pass
                elif rgbsigma.shape[0] == 3:
                    # Only 3 channels, add a dummy density channel
                    print("Adding dummy density channel")
                    density = torch.ones((1, *rgbsigma.shape[1:]), dtype=torch.float32)
                    rgbsigma = torch.cat([rgbsigma, density], dim=0)
                elif rgbsigma.shape[0] > 4:
                    # More than 4 channels, take first 4
                    print(f"Taking first 4 channels from {rgbsigma.shape[0]} channels")
                    rgbsigma = rgbsigma[:4]
            elif len(rgbsigma.shape) == 5:  # [B, C, H, W, D] format
                if rgbsigma.shape[1] == 4:
                    # Already has 4 channels, perfect
                    pass
                elif rgbsigma.shape[1] == 3:
                    # Only 3 channels, add a dummy density channel
                    print("Adding dummy density channel")
                    density = torch.ones(
                        (rgbsigma.shape[0], 1, *rgbsigma.shape[2:]), dtype=torch.float32
                    )
                    rgbsigma = torch.cat([rgbsigma, density], dim=1)
                elif rgbsigma.shape[1] > 4:
                    # More than 4 channels, take first 4
                    print(f"Taking first 4 channels from {rgbsigma.shape[1]} channels")
                    rgbsigma = rgbsigma[:, :4]

            # Add batch dimension if not present
            if len(rgbsigma.shape) == 4:  # [C, H, W, D]
                rgbsigma = rgbsigma.unsqueeze(0)  # [1, C, H, W, D]

            # Check if we need to resize to 160³
            target_size = (160, 160, 160)
            current_size = rgbsigma.shape[2:]

            if current_size != target_size:
                print(f"Resizing grid from {current_size} to {target_size}")

                # Create resized tensor using trilinear interpolation
                resized = F.interpolate(
                    rgbsigma, size=target_size, mode="trilinear", align_corners=False
                )

                rgbsigma = resized

            print(f"Final tensor shape: {rgbsigma.shape}")
            # Final shape should be [1, 4, 160, 160, 160]

            return rgbsigma.to(device)

        except Exception as e:
            print(f"Error in fallback loading: {e}")
            import traceback

            traceback.print_exc()
            raise


def render_views(radiance_field, n_views=20):
    """
    Render views of a radiance field from different angles

    Args:
        radiance_field: Radiance field tensor [B, 4, H, W, D]
        n_views: Number of views to render

    Returns:
        rendered_views: List of rendered views [n_views, H, W, 3]
    """
    # Get viewing angles from fibonacci sphere
    view_points = fibonacci_sphere(n_views)

    # Placeholder for views
    views = []

    # Get dimensions - should be [1, 4, 160, 160, 160] at this point
    grid_size_h = radiance_field.shape[2]
    grid_size_w = radiance_field.shape[3]
    grid_size_d = radiance_field.shape[4]

    print(f"Radiance field dimensions: {grid_size_h} x {grid_size_w} x {grid_size_d}")

    # For MS-SSIM, output images need to be larger than 160 pixels
    # We'll target 224x224 to ensure we're safely above this threshold
    target_size = (224, 224)

    for x, y, z in tqdm(view_points, desc="Rendering views"):
        # Calculate view direction from the point on the sphere
        view_dir = np.array([x, y, z])
        view_dir = view_dir / np.linalg.norm(view_dir)

        # Instead of taking slices, we'll render full projections
        # We'll create max-projections along each axis

        if abs(view_dir[0]) > abs(view_dir[1]) and abs(view_dir[0]) > abs(view_dir[2]):
            # X is dominant axis - project along X
            rgb_volume = radiance_field[0, :3, :, :, :]  # [3, H, W, D]

            # Create a weighted average projection along X axis
            # Weight by density channel to simulate volume rendering
            density = radiance_field[0, 3:4, :, :, :]  # [1, H, W, D]
            weights = F.softmax(density * 10, dim=1)  # Scale for sharper weighting
            view_slice = torch.sum(rgb_volume * weights, dim=1)  # [3, W, D]
            # Ensure consistent output format [H, W, 3]
            view_slice = view_slice.permute(1, 2, 0)  # [W, D, 3]
        elif abs(view_dir[1]) > abs(view_dir[2]):
            # Y is dominant axis - project along Y
            rgb_volume = radiance_field[0, :3, :, :, :]  # [3, H, W, D]

            # Create a weighted average projection along Y axis
            density = radiance_field[0, 3:4, :, :, :]  # [1, H, W, D]
            weights = F.softmax(density * 10, dim=2)  # Scale for sharper weighting
            view_slice = torch.sum(rgb_volume * weights, dim=2)  # [3, H, D]

            # Ensure consistent output format [H, W, 3]
            view_slice = view_slice.permute(1, 2, 0)  # [H, D, 3]
        else:
            # Z is dominant axis - project along Z
            rgb_volume = radiance_field[0, :3, :, :, :]  # [3, H, W, D]

            # Create a weighted average projection along Z axis
            density = radiance_field[0, 3:4, :, :, :]  # [1, H, W, D]
            weights = F.softmax(density * 10, dim=3)  # Scale for sharper weighting
            view_slice = torch.sum(rgb_volume * weights, dim=3)  # [3, H, W]

            # Ensure consistent output format [H, W, 3]
            view_slice = view_slice.permute(1, 2, 0)  # [H, W, 3]

        # Apply any post-processing
        view_slice = torch.clamp(view_slice, 0, 1)

        # Double-check that our tensor has the right shape before resizing
        print(f"View slice shape before resizing: {view_slice.shape}")
        # Ensure we have 3 channels in the last dimension
        if view_slice.shape[-1] != 3:
            print(
                f"Warning: View has {view_slice.shape[-1]} channels instead of 3. Fixing..."
            )
            # If it's a [H, W, C] with C != 3, keep only the first 3 channels if C > 3
            if view_slice.shape[-1] > 3:
                view_slice = view_slice[..., :3]
            # If it's less than 3 channels, we need to expand it
            elif view_slice.shape[-1] == 1:
                view_slice = view_slice.expand(-1, -1, 3)
            else:
                # If it's in a completely different format, try to reshape
                view_slice = view_slice.reshape(
                    view_slice.shape[0], view_slice.shape[1], -1
                )
                if view_slice.shape[-1] > 3:
                    view_slice = view_slice[..., :3]
                elif view_slice.shape[-1] < 3:
                    view_slice = torch.cat(
                        [
                            view_slice,
                            torch.zeros_like(
                                view_slice[..., : 3 - view_slice.shape[-1]]
                            ),
                        ],
                        dim=-1,
                    )
            print(f"Fixed to shape: {view_slice.shape}")

        # Resize to target size to ensure metrics can be calculated
        view_slice = view_slice.permute(2, 0, 1)  # [3, H, W]
        view_slice = F.interpolate(
            view_slice.unsqueeze(0),  # Add batch dim
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )[
            0
        ]  # Remove batch dim
        view_slice = view_slice.permute(1, 2, 0)  # [H, W, 3]

        # Final check to ensure correct shape
        if view_slice.shape[-1] != 3:
            print(
                f"Error: View still doesn't have 3 channels after resizing: {view_slice.shape}"
            )
            continue

        views.append(view_slice)

        # Print debug info for the first view
        if len(views) == 1:
            print(f"First view shape: {view_slice.shape}")

    return views


def save_views_to_directory(views, output_dir, prefix="view"):
    """
    Save rendered views to a directory

    Args:
        views: List of rendered views as tensors [H, W, 3]
        output_dir: Directory to save views to
        prefix: Prefix for filenames
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Saving {len(views)} views to {output_dir}")

    for i, view in enumerate(views):
        try:
            # Convert tensor to numpy array
            img_tensor = view.cpu()

            # Debug the tensor shape
            print(f"View {i} tensor shape before conversion: {img_tensor.shape}")

            # Ensure the tensor has the shape [H, W, 3]
            if len(img_tensor.shape) == 3 and img_tensor.shape[2] == 3:
                # Already in the correct format [H, W, 3]
                pass
            elif len(img_tensor.shape) == 3 and img_tensor.shape[0] == 3:
                # Format is [3, H, W], convert to [H, W, 3]
                img_tensor = img_tensor.permute(1, 2, 0)
                print(f"Permuted tensor to shape: {img_tensor.shape}")
            else:
                # If we got here, the tensor has an unexpected shape
                print(
                    f"Warning: Skipping view {i} due to invalid shape {img_tensor.shape}"
                )
                continue

            # Convert to numpy and ensure RGB values are in range [0, 1] for matplotlib
            img_np = img_tensor.numpy()
            img_np = np.clip(img_np, 0, 1)

            # Final check of image shape
            print(f"Final image shape for view {i}: {img_np.shape}")

            if img_np.shape[-1] != 3:
                print(
                    f"Error: Cannot save view {i} with shape {img_np.shape}, expected 3 channels"
                )
                continue

            # Save image
            filename = f"{prefix}_{i:03d}.png"
            output_file = output_path / filename

            # Use PIL for more reliable image saving
            img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
            img_pil.save(str(output_file))
            print(f"Saved view {i} to {output_file}")

        except Exception as e:
            print(f"Error saving view {i}: {e}")

    print(f"Saved views to {output_dir}")


def calculate_metrics(gt_views, gen_views):
    """
    Calculate image quality metrics between ground truth and generated views.

    Args:
        gt_views: List of ground truth views
        gen_views: List of generated views

    Returns:
        metrics: Dictionary of metrics
    """
    psnr_values = []
    ms_ssim_values = []
    lpips_values = []

    # Configure the metrics
    # Data range is 1.0 since our images are in [0, 1]
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

        # Verify both images have the same shape
        if gt_img.shape != gen_img.shape:
            print(
                f"Warning: Image shapes don't match for view {i}. Resizing generated image to match ground truth."
            )
            # Resize the generated image to match ground truth
            gen_img = gen_img.permute(2, 0, 1)  # [C, H, W]
            gen_img = F.interpolate(
                gen_img.unsqueeze(0),  # Add batch dim
                size=(gt_img.shape[0], gt_img.shape[1]),
                mode="bilinear",
                align_corners=False,
            )[
                0
            ]  # Remove batch dim
            gen_img = gen_img.permute(1, 2, 0)  # [H, W, C]

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
            # Convert images to format required by torchmetrics (B, C, H, W)
            gt_psnr = gt_img.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
            gen_psnr = gen_img.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]

            # Check if images are nearly identical
            # mse = torch.mean((gt_img - gen_img) ** 2)
            # if mse < 1e-10:
            #     print(f"Images for view {i} are nearly identical, setting PSNR to 100.0")
            #     psnr = torch.tensor(100.0, device=gt_img.device)
            # else:
            # Use torchmetrics' PSNR implementation
            psnr = peak_signal_noise_ratio(
                gen_psnr,
                gt_psnr,
                data_range=1.0,  # Since our images are in [0, 1] range
                dim=(1, 2, 3),  # Average over C, H, W dimensions
            )

            psnr_values.append(psnr.item())
        except Exception as e:
            print(f"Error calculating PSNR for view {i}: {e}")

        # For MS-SSIM
        try:
            # Convert from (H, W, C) to (1, C, H, W)
            gt_ssim = gt_img.permute(2, 0, 1).unsqueeze(0)
            gen_ssim = gen_img.permute(2, 0, 1).unsqueeze(0)

            # Verify images are large enough for MS-SSIM
            if gt_ssim.shape[2] <= 160 or gt_ssim.shape[3] <= 160:
                print(
                    f"Warning: Images too small for MS-SSIM (min 161 pixels required). View {i} shape: {gt_ssim.shape}"
                )
                continue

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
            # Normalize to [-1, 1] range as required by LPIPS
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


def compare_radiance_fields(path1, path2, n_views=20, output_dir=None):
    """
    Compare two radiance fields by rendering views and calculating metrics

    Args:
        path1: Path to the first radiance field
        path2: Path to the second radiance field
        n_views: Number of views to render
        output_dir: Directory to save rendered views (optional)

    Returns:
        metrics: Dictionary of metrics
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load radiance fields
    print(f"Loading radiance field 1 from {path1}")
    rf1 = load_radiance_field(path1, device)

    print(f"Loading radiance field 2 from {path2}")
    rf2 = load_radiance_field(path2, device)

    # Render views
    print(f"Rendering {n_views} views for radiance field 1")
    views1 = render_views(rf1, n_views)

    print(f"Rendering {n_views} views for radiance field 2")
    views2 = render_views(rf2, n_views)

    # Save views if output directory is provided
    if output_dir:
        # Create base output directory
        base_dir = Path(output_dir)

        # Create subdirectories for each radiance field
        ref_dir = base_dir / "reference"
        target_dir = base_dir / "target"

        # Extract filenames from paths for more descriptive output
        path1_name = Path(path1).stem
        path2_name = Path(path2).stem

        # Save views
        save_views_to_directory(views1, ref_dir, prefix=f"{path1_name}_view")
        save_views_to_directory(views2, target_dir, prefix=f"{path2_name}_view")

    # Calculate metrics
    print("Calculating metrics between views")
    metrics = calculate_metrics(views1, views2)

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Calculate metrics between two radiance fields"
    )

    parser.add_argument(
        "--path1",
        type=str,
        required=True,
        help="Path to the first radiance field NPZ file",
    )
    parser.add_argument(
        "--path2",
        type=str,
        required=True,
        help="Path to the second radiance field NPZ file",
    )
    parser.add_argument(
        "--views", type=int, default=20, help="Number of views to render"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save rendered views (optional)",
    )

    args = parser.parse_args()

    # Compare radiance fields
    metrics = compare_radiance_fields(
        args.path1, args.path2, args.views, args.output_dir
    )

    # Print metrics
    print("\nMetrics between radiance fields:")
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


if __name__ == "__main__":
    main()
