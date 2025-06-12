import os
import argparse
import torch
from PIL import Image
from default_config import cfg as config
from models.lion import LION
from utils.vis_helper import plot_points


def save_as_obj(points, filename):
    """Save points as OBJ file"""
    with open(filename, "w") as f:
        for point in points:
            f.write(f"v {point[0]} {point[1]} {point[2]}\n")


def save_as_ply(points, filename):
    """Save points as PLY file"""
    with open(filename, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for point in points:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")


def main():
    parser = argparse.ArgumentParser(description="Generate 3D point clouds using LION")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./lion_ckpt/unconditional",
        help="Directory containing the checkpoints",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="chair",
        choices=["chair", "airplane", "car", "all55"],
        help="Category to generate",
    )
    parser.add_argument(
        "--num_samples", type=int, default=5, help="Number of samples to generate"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./generated_samples",
        help="Output directory for generated point clouds",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="pt",
        choices=["pt", "obj", "ply"],
        help="Output format for the point clouds",
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Generate visualization images"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)

    # Set paths based on category
    model_path = os.path.join(
        args.checkpoint_dir,
        args.category,
        "checkpoints",
        "epoch_10999_iters_2100999.pt",
    )
    config_path = os.path.join(args.checkpoint_dir, args.category, "cfg.yml")

    # Check if files exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    # Create output directory
    output_dir = os.path.join(args.output_dir, args.category)
    os.makedirs(output_dir, exist_ok=True)

    # Load config
    print(f"Loading config from {config_path}")
    config.merge_from_file(config_path)

    # Initialize model
    print(f"Initializing LION model for category: {args.category}")
    lion = LION(config)
    print(f"Loading checkpoint from {model_path}")
    lion.load_model(model_path)

    print(f"Generating {args.num_samples} samples...")

    @torch.no_grad()
    def sample(self, num_samples=10, clip_feat=None, save_img=False):
        self.scheduler.set_timesteps(1000, device="cuda")
        timesteps = self.scheduler.timesteps
        latent_shape = self.vae.latent_shape()
        global_prior, local_prior = self.priors[0], self.priors[1]
        assert not local_prior.mixed_prediction and not global_prior.mixed_prediction
        sampled_list = []
        output_dict = {}

        # start sample global prior
        x_T_shape = [num_samples] + latent_shape[0]
        x_noisy = torch.randn(size=x_T_shape, device="cuda")
        condition_input = None
        for i, t in enumerate(timesteps):
            t_tensor = torch.ones(num_samples, dtype=torch.int64, device="cuda") * (
                t + 1
            )
            noise_pred = global_prior(
                x=x_noisy,
                t=t_tensor.float(),
                condition_input=condition_input,
                clip_feat=clip_feat,
            )
            x_noisy = self.scheduler.step(noise_pred, t, x_noisy).prev_sample
        sampled_list.append(x_noisy)
        output_dict["z_global"] = x_noisy
        print(f"z_global: {x_noisy.shape}")

        condition_input = x_noisy
        condition_input = self.vae.global2style(condition_input)

        # start sample local prior
        x_T_shape = [num_samples] + latent_shape[1]
        x_noisy = torch.randn(size=x_T_shape, device="cuda")

        for i, t in enumerate(timesteps):
            t_tensor = torch.ones(num_samples, dtype=torch.int64, device="cuda") * (
                t + 1
            )
            noise_pred = local_prior(
                x=x_noisy,
                t=t_tensor.float(),
                condition_input=condition_input,
                clip_feat=clip_feat,
            )
            x_noisy = self.scheduler.step(noise_pred, t, x_noisy).prev_sample
        sampled_list.append(x_noisy)
        output_dict["z_local"] = x_noisy
        print(f"z_local: {x_noisy.shape}")
        print(f"sampled_list: {len(sampled_list)}")

        # decode the latent
        output = self.vae.sample(num_samples=num_samples, decomposed_eps=sampled_list)
        if save_img:
            out_name = plot_points(output, "/tmp/tmp.png")
            print(f"INFO save plot image at {out_name}")
        output_dict["points"] = output
        return output_dict, sampled_list

    # Generate samples
    output, sampled_list = sample(lion, args.num_samples)
    pts = output["points"]

    print(f"Samples generated. Saving results to {output_dir}")

    # Save results
    print(pts.shape)
    for i in range(args.num_samples):
        sample_points = pts[i].cpu().numpy()
        base_filename = os.path.join(output_dir, f"sample_{i}")

        # Save in the requested format
        if args.format == "pt":
            torch.save(pts[i], f"{base_filename}.pt")
        elif args.format == "obj":
            save_as_obj(sample_points, f"{base_filename}.obj")
        elif args.format == "ply":
            save_as_ply(sample_points, f"{base_filename}.ply")

        # Generate visualization if requested
        if args.visualize:
            plot_points(pts[i : i + 1], output_name=f"{base_filename}.png")

        print(f"Saved sample {i} to {base_filename}.{args.format}")

    print(f"Successfully generated {args.num_samples} samples for {args.category}!")

    output, sampled_list = sample(lion, args.num_samples)
    pts = output["points"]

    print(f"Samples generated. Saving results to {output_dir}")

    # Save results
    print(pts.shape)
    for i in range(args.num_samples):
        sample_points = pts[i].cpu().numpy()
        base_filename = os.path.join(output_dir, f"sample_{i+1}")

        # Save in the requested format
        if args.format == "pt":
            torch.save(pts[i], f"{base_filename}.pt")
        elif args.format == "obj":
            save_as_obj(sample_points, f"{base_filename}.obj")
        elif args.format == "ply":
            save_as_ply(sample_points, f"{base_filename}.ply")

        # Generate visualization if requested
        if args.visualize:
            plot_points(pts[i : i + 1], output_name=f"{base_filename}.png")

        print(f"Saved sample {i} to {base_filename}.{args.format}")

    print(f"Successfully generated {args.num_samples} samples for {args.category}!")

    # with torch.no_grad():
    #     pts = lion.vae.sample(args.num_samples, decomposed_eps=sampled_list)
    # for i in range(args.num_samples):
    #     sample_points = pts[i].cpu().numpy()
    #     base_filename = os.path.join(output_dir, f'sample_{i}')

    #     # Save in the requested format
    #     if args.format == 'pt':
    #         torch.save(pts[i], f"{base_filename}.pt")
    #     elif args.format == 'obj':
    #         save_as_obj(sample_points, f"{base_filename}.obj")
    #     elif args.format == 'ply':
    #         save_as_ply(sample_points, f"{base_filename}_2.ply")

    # Show a summary of what was generated
    if args.visualize:
        first_img_path = os.path.join(output_dir, "sample_0.png")
        print(f"Visualizations saved as PNG files. First sample: {first_img_path}")

        # Optionally display the first image
        try:
            img = Image.open(first_img_path)
            img.show()
        except Exception as e:
            print(f"Note: Could not display image: {e}")


if __name__ == "__main__":
    main()
