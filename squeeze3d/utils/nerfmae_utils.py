import torch
from ..nerf_mae.model.mae.swin_mae3d import SwinTransformer_MAE3D_New
import numpy as np


def load_model(
    model_path="/home/rishit/streaming-nerf/weights/NeRF-MAE/nerf_mae_pretrained.pt",
    backbone_type="swin_s",
    device=None,
):
    swin_config = {
        "swin_t": {
            "embed_dim": 96,
            "depths": [2, 2, 6, 2],
            "num_heads": [3, 6, 12, 24],
        },
        "swin_s": {
            "embed_dim": 96,
            "depths": [2, 2, 18, 2],
            "num_heads": [3, 6, 12, 24],
        },
        "swin_b": {
            "embed_dim": 128,
            "depths": [2, 2, 18, 2],
            "num_heads": [3, 6, 12, 24],
        },
        "swin_l": {
            "embed_dim": 192,
            "depths": [2, 2, 18, 2],
            "num_heads": [6, 12, 24, 48],
        },
    }

    config = swin_config[backbone_type]

    model = SwinTransformer_MAE3D_New(
        patch_size=[4, 4, 4],
        embed_dim=config["embed_dim"],
        depths=config["depths"],
        num_heads=config["num_heads"],
        window_size=[4, 4, 4],
        stochastic_depth_prob=0.1,
        expand_dim=True,
        resolution=160,
    )

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    model.eval()

    return model, device


def density_to_alpha(density):
    return np.clip(1.0 - np.exp(-np.exp(density) / 100.0), 0.0, 1.0)


def load_data(features_path, resolution=160, normalize_density=True):
    # print(f"Loading features from {features_path}")
    feature = np.load(features_path, allow_pickle=True)

    res = feature["resolution"]
    rgbsigma = feature["rgbsigma"]

    # print(f"Original resolution: {res}")
    # print(f"Original rgbsigma shape: {rgbsigma.shape}")

    if normalize_density:
        alpha = density_to_alpha(rgbsigma[..., -1])
        rgbsigma[..., -1] = alpha

    rgbsigma = np.transpose(rgbsigma, (3, 0, 1, 2))
    rgbsigma = torch.from_numpy(rgbsigma)

    if rgbsigma.dtype == torch.uint8:
        print("Converting uint8 values to float in range [0, 1]")
        rgbsigma = rgbsigma.float() / 255.0

    if len(rgbsigma.shape) == 4:
        rgbsigma = rgbsigma.unsqueeze(0)

    if rgbsigma.shape[2:] != (resolution, resolution, resolution):
        # print(
        #     f"Resizing from {rgbsigma.shape[2:]} to ({resolution}, {resolution}, {resolution})"
        # )
        rgbsigma = torch.nn.functional.interpolate(
            rgbsigma,
            size=(resolution, resolution, resolution),
            mode="trilinear",
            align_corners=False,
        )

    # print(f"Processed rgbsigma shape: {rgbsigma.shape}")

    return rgbsigma


def save_npz_radiance_field(radiance_field, input_path, output_path):
    """
    Save a radiance field to a NPZ file using the same format as the input file.

    Args:
        radiance_field: The radiance field tensor to save
        input_path: Path to the original input NPZ file (to match structure)
        output_path: Path to save the output NPZ file

    Returns:
        bool: Success flag
    """
    print(f"Saving radiance field to NPZ at {output_path}...")

    if isinstance(radiance_field, torch.Tensor):
        radiance_field_np = radiance_field.detach().cpu().numpy()
    else:
        radiance_field_np = radiance_field

    original_data = np.load(input_path, allow_pickle=True)

    output_data = {}

    if "rgbsigma" in original_data:
        print("Using 'rgbsigma' key format from original file")

        if len(radiance_field_np.shape) == 5:
            if radiance_field_np.shape[0] == 1:
                radiance_field_np = radiance_field_np[0]

            if len(radiance_field_np.shape) == 4 and radiance_field_np.shape[0] <= 4:
                print(f"Transposing from [C, H, W, D] to [H, W, D, C]")
                radiance_field_np = np.transpose(radiance_field_np, (1, 2, 3, 0))

        output_data["rgbsigma"] = radiance_field_np

        if "resolution" in original_data:
            output_data["resolution"] = original_data["resolution"]

    elif "features" in original_data:
        print("Using 'features' key format from original file")

        if len(radiance_field_np.shape) == 5:
            if radiance_field_np.shape[0] == 1:
                radiance_field_np = radiance_field_np[0]

            if len(radiance_field_np.shape) == 4 and radiance_field_np.shape[0] <= 4:
                print(f"Transposing from [C, H, W, D] to [H, W, D, C]")
                radiance_field_np = np.transpose(radiance_field_np, (1, 2, 3, 0))

        output_data["features"] = radiance_field_np

    else:
        key = list(original_data.keys())[0]
        print(f"Using '{key}' key format from original file")

        if len(radiance_field_np.shape) == 5:
            if radiance_field_np.shape[0] == 1:
                radiance_field_np = radiance_field_np[0]

        output_data[key] = radiance_field_np

    np.savez_compressed(output_path, **output_data)
    print(f"Successfully saved radiance field to {output_path}")
    print(f"Saved data keys: {list(output_data.keys())}")
    return True


@torch.inference_mode()
def encode_to_latent(input_grid, model, device):
    input_grid = input_grid.to(device)

    with torch.no_grad():
        if input_grid.shape[2:] != (160, 160, 160):
            print(
                f"Input grid shape {input_grid.shape} doesn't match expected (B, 4, 160, 160, 160)"
            )
            input_grid = torch.nn.functional.interpolate(
                input_grid, size=(160, 160, 160), mode="trilinear", align_corners=False
            )

        x = model.patch_partition(input_grid)
        x = (
            x
            + model.pos_embed.type_as(input_grid).to(input_grid.device).clone().detach()
        )

        multi_scale_features = []

        for i, stage in enumerate(model.stages):
            x = stage(x)
            # Store feature map in standard format [B, C, H, W, D]
            # Original x is in [B, H, W, D, C] format
            multi_scale_features.append(torch.permute(x, [0, 4, 1, 2, 3]).contiguous())

        dec3 = model.decoder4(multi_scale_features[3], multi_scale_features[2])
        dec2 = model.decoder3(dec3, multi_scale_features[1])
        latent_representation = model.decoder2(dec2, multi_scale_features[0])

    # print("\nLatent Space Representation:")
    # print(f"Final latent shape: {latent_representation.shape}")
    # print("Multi-scale feature shapes:")
    # for i, feat in enumerate(multi_scale_features):
    #     print(f"Level {i+1}: {feat.shape}")

    return latent_representation


@torch.inference_mode()
def decode_from_latent(latent_representation, model, device):
    with torch.no_grad():
        dec0 = model.decoder1(latent_representation)

        # Final output projection to get the reconstructed grid
        reconstructed_grid = model.out(dec0)

    return reconstructed_grid


if __name__ == "__main__":
    model, device = load_model()
    input_grid = load_data(
        "/scratch/rishit/NeRF-MAE/pretrain/features/3dfront_2014_00.npz"
    )
    latent_representation = encode_to_latent(input_grid, model, device)
    reconstructed_grid = decode_from_latent(latent_representation, model, device)
    print(latent_representation.shape)
    print(reconstructed_grid.shape)
