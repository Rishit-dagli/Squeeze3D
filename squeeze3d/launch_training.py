import math
import os
import re
import tempfile
from typing import List, Union

import lpips
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset, load_from_disk
from PIL import Image
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb

from .utils.shape_utils import decode, load_shape_pipeline
from .utils.lrm_utils import infer_lrm, load_lrm, lrm_decode, lrm_encode
from .models.conv import HighDimensionalUNet, ThreeLayerConvMLP, TwoLayerCNN
from .models.mlp import (
    LT,
    ConstMLP,
    LT2Transformer,
    LTOrtho,
    ResidualNN,
    SimpleProjection,
    ThreeLayerMLP,
    TwoLayerMLP,
    LTpc,
    LTNeRF,
    LTpcOrtho,
    LTNeRFOrtho,
)
from .optimizers.muon import Muon
from .optimizers.schedule_free import AdamWScheduleFreeClosure
from .optimizers.soap import SOAP
from .utils.utils import seed_all
from .utils.zero123_utils import generate_multiview_outputs as generate_zero123_outputs
from .utils.zero123_utils import get_latents as get_zero123_latents
from .utils.zero123_utils import load_model as load_zero123_model


def compute_kl_div(pred, target):
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)
    pred_probs = F.softmax(pred, dim=-1)
    target_probs = F.softmax(target, dim=-1)
    kl_div = F.kl_div(pred_probs.log(), target_probs, reduction="batchmean")
    return kl_div


def visualize_multiview_outputs(multiview_outputs):
    num_views = len(multiview_outputs)
    fig, axes = plt.subplots(1, num_views, figsize=(num_views * 3, 3))

    for i, output in enumerate(multiview_outputs):
        output_image = output.squeeze(0).permute(1, 2, 0).cpu().numpy()

        output_image = (output_image + 1) / 2.0
        output_image = np.clip(output_image, 0, 1)

        axes[i].imshow(output_image)
        axes[i].axis("off")

    return fig


def compute_photometric_loss(image1, image2):
    loss = F.mse_loss(image1, image2)
    return loss


def ortho_loss(criterion, outputs, targets, b):
    primary_loss = criterion(outputs, targets)

    b_normalized = b / (torch.norm(b, dim=1, keepdim=True) + 1e-8)
    gram_matrix = torch.matmul(b_normalized, b_normalized.transpose(0, 1))
    target = torch.eye(b.shape[0], device=b.device)
    loss = torch.mean((gram_matrix - target) ** 2)

    # batch_size = 1024
    # ortho_penalty = 0

    # for i in range(0, norm_outputs.shape[1], batch_size):
    #     outputs_chunk = norm_outputs[:, i : i + batch_size]
    #     similarity = outputs_chunk.T @ outputs_chunk
    #     identity = torch.eye(similarity.shape[0], device=outputs.device)
    #     ortho_penalty += criterion(similarity, identity)

    # print(f"Primary Loss: {primary_loss.item()}, Ortho Loss: {loss.item()}")

    return primary_loss + loss * 1e7


def pil_images_to_batch(
    images: List[Image.Image], size: Union[int, tuple] = None
) -> torch.Tensor:
    if not images:
        raise ValueError("Image list cannot be empty")
    transform_list = []
    if size:
        if isinstance(size, int):
            size = (size, size)
        transform_list.append(transforms.Resize(size))

    transform_list.extend(
        [
            transforms.ToTensor(),
        ]
    )

    transform = transforms.Compose(transform_list)
    try:
        tensor_list = [transform(img) for img in images]
        batch = torch.stack(tensor_list, dim=0)
        return batch
    except RuntimeError as e:
        raise ValueError("All images must have the same size and mode") from e


def calculate_batch_lpips(batch1: torch.Tensor, batch2: torch.Tensor) -> torch.Tensor:
    lpips_loss = lpips.LPIPS(net="alex")
    distances = lpips_loss(batch1, batch2)
    return distances


def normalize_batch_for_lpips(batch: torch.Tensor) -> torch.Tensor:
    return batch * 2 - 1


def cosine_similarity_loss(pred, target):
    pred_norm = F.normalize(pred, p=2, dim=1)
    target_norm = F.normalize(target, p=2, dim=1)

    similarity = nn.CosineSimilarity(dim=1)(pred_norm, target_norm)
    return 1.0 - similarity.mean()


def create_dataloaders(
    train_dataset,
    val_dataset,
    batch_size,
    pin_memory=True,
    num_workers=min(os.cpu_count(), 12),
):
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
    )

    return train_dataloader, val_dataloader


def compute_grad_norm(model, norm_type=1.0):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1.0 / norm_type)
    return total_norm


def compute_stft(tensor, n_fft=400, hop_length=160, win_length=400):
    return torch.stft(
        tensor,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        return_complex=True,
    )


def compute_spectral_loss(output, target):
    output_stft = compute_stft(output)
    target_stft = compute_stft(target)
    return F.l1_loss(output_stft, target_stft)


def train_epoch(
    model,
    shape_model,
    dataloader,
    optimizer,
    scheduler,
    criterion,
    accelerator,
    gradient_accumulation_steps,
    clip_grad,
    max_grad_norm,
    epoch,
    use_spectral_loss,
    use_decoder_supervision,
    prefix,
    plot_gifs=True,
    zero123_model=None,
    lrm_model=None,
    decoder_model="shape",
    use_ortho_loss=False,
):
    model.train()
    total_loss = 0
    for i, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False)):
        if decoder_model == "instantmesh_triplane":
            inputs, targets = batch["inputs"], batch["planes"]
        elif decoder_model == "nerf":
            # For NeRF, both inputs and targets come from the "latent" column
            inputs = batch["latent"]
            inputs = inputs.squeeze(1)
            targets = inputs.clone()  # Create a separate copy for targets
        else:
            inputs, targets = batch["inputs"], batch["outputs"]
            if decoder_model == "pc":
                inputs = inputs.squeeze(2)
                inputs = inputs.squeeze(2)
                inputs, targets = targets, inputs

        with accelerator.accumulate(model):
            if isinstance(model, (LTOrtho, LTpcOrtho, LTNeRFOrtho)):
                outputs, b = model(inputs)
            else:
                outputs = model(inputs)

            if zero123_model:
                if i == 0 and epoch % 25 == 0 and plot_gifs:
                    with torch.no_grad():
                        multiview_outputs = generate_zero123_outputs(
                            zero123_model, outputs
                        )
                        multiview_gt = generate_zero123_outputs(zero123_model, targets)
                        wandb.log(
                            {
                                "train/multiview_outputs": wandb.Image(
                                    visualize_multiview_outputs(multiview_outputs)
                                ),
                                "train/multiview_gt": wandb.Image(
                                    visualize_multiview_outputs(multiview_gt)
                                ),
                            }
                        )
            elif decoder_model == "shape":
                if (i == 0 and epoch % 25 == 0 and plot_gifs) or (
                    use_decoder_supervision and i == 0 and epoch % 5 == 0
                ):
                    prompt = [""]
                    with (
                        tempfile.NamedTemporaryFile(
                            suffix=".gif"
                        ) as ground_truth_gif_file,
                        tempfile.NamedTemporaryFile(
                            suffix=".gif"
                        ) as generated_gif_file,
                    ):

                        with torch.no_grad():
                            ground_truth_gif_path = ground_truth_gif_file.name
                            generated_gif_path = generated_gif_file.name
                            ground_truth_gif_filename = ground_truth_gif_path.replace(
                                ".gif", ""
                            )
                            generated_gif_filename = generated_gif_path.replace(
                                ".gif", ""
                            )

                            _ = decode(
                                shape_model,
                                targets[0].unsqueeze(0).half(),
                                prompt=prompt,
                                gif=True,
                                mesh=False,
                                export_to=[ground_truth_gif_filename],
                                return_trimesh=False,
                            ).images[0]
                            _ = decode(
                                shape_model,
                                outputs[0].unsqueeze(0).half(),
                                prompt=prompt,
                                gif=True,
                                mesh=False,
                                export_to=[generated_gif_filename],
                                return_trimesh=False,
                            ).images[0]

                        wandb.log(
                            {
                                "train/ground_truth_gif": wandb.Video(
                                    ground_truth_gif_path, format="gif"
                                ),
                                "train/generated_gif": wandb.Video(
                                    generated_gif_path, format="gif"
                                ),
                            }
                        )
            elif decoder_model == "lrm":
                if (i == 5 and epoch % 10 == 0 and plot_gifs) or (
                    use_decoder_supervision and i == 0 and epoch % 5 == 0
                ):
                    with (
                        tempfile.TemporaryDirectory() as gt_dir,
                        tempfile.TemporaryDirectory() as gen_dir,
                    ):
                        with torch.no_grad():
                            # print shapes
                            # print(targets[0].unsqueeze(0).shape, outputs[0].unsqueeze(0).shape)

                            ground_truth_gif_path, _ = lrm_decode(
                                targets[0].unsqueeze(0), lrm_model, True, True, gt_dir
                            )

                            try:
                                generated_gif_path, _ = lrm_decode(
                                    outputs[0].unsqueeze(0).view(1, 3, 48, 64, 64),
                                    lrm_model,
                                    True,
                                    True,
                                    gen_dir,
                                )
                            except IndexError:
                                generated_gif_path = None
                                print("Could not render gif")

                            # copy mp4 to current directory
                            # import shutil
                            # shutil.copy(ground_truth_gif_path, os.path.join(os.getcwd(), "ground_truth.mp4"))

                        if generated_gif_path:
                            wandb.log(
                                {
                                    "train/generated_gif": wandb.Video(
                                        generated_gif_path, format="mp4"
                                    ),
                                }
                            )
                        wandb.log(
                            {
                                "train/ground_truth_gif": wandb.Video(
                                    ground_truth_gif_path, format="mp4"
                                ),
                            }
                        )

            if use_spectral_loss:
                # in case of lrm do a view for the outputs
                if use_spectral_loss == "kld":
                    loss = compute_kl_div(outputs, targets.view(outputs.shape))
                else:
                    loss = compute_spectral_loss(outputs, targets.view(outputs.shape))
            elif use_decoder_supervision:
                with torch.no_grad():
                    generated_images = decode(
                        shape_model,
                        outputs.half(),
                        prompt=[""] * int(outputs.shape[0]),
                        gif=True,
                        mesh=False,
                        export_to=None,
                        return_trimesh=False,
                    ).images

                    target_images = decode(
                        shape_model,
                        targets.half(),
                        prompt=[""] * int(outputs.shape[0]),
                        gif=True,
                        mesh=False,
                        export_to=None,
                        return_trimesh=False,
                    ).images

                generated_tensor = pil_images_to_batch(generated_images[0]).to(
                    outputs.device
                )
                target_tensor = pil_images_to_batch(target_images[0]).to(outputs.device)

                generated_tensor = generated_tensor.detach().requires_grad_(True)

                loss = compute_photometric_loss(generated_tensor, target_tensor)

                loss = loss * outputs.sum() * 0 + loss
            elif use_ortho_loss:
                loss = ortho_loss(criterion, outputs, targets.view(outputs.shape), b)
            else:
                # Ensure proper reshaping for targets based on outputs shape
                if decoder_model == "nerf":
                    # For NeRF, ensure the targets are properly flattened to match outputs
                    loss = criterion(outputs, targets.reshape(outputs.shape))
                else:
                    # For other models, use the existing logic
                    loss = criterion(outputs, targets.view(outputs.shape))

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                if clip_grad:
                    accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)

                l1_grad_norm = compute_grad_norm(model, norm_type=1.0)
                l2_grad_norm = compute_grad_norm(model, norm_type=2.0)

                if prefix:
                    with torch.no_grad():
                        mse = criterion(outputs, targets)
                    accelerator.log(
                        {
                            f"train/{str(prefix)}/loss": loss.item(),
                            f"train/{str(prefix)}/mse": mse.item(),
                        }
                    )
                else:
                    accelerator.log(
                        {
                            "train/l1_grad_norm": l1_grad_norm,
                            "train/l2_grad_norm": l2_grad_norm,
                            "train/loss": loss.item(),
                            "train/lr": scheduler.get_last_lr()[0],
                        }
                    )

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(
    model,
    shape_model,
    dataloader,
    criterion,
    accelerator,
    desc="Validation",
    epoch=0,
    plot_gifs=True,
    use_decoder_supervision=False,
    zero123_model=None,
    lrm_model=None,
    decoder_model="shape",
    use_ortho_loss=False,
):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc=desc, leave=False)):
            if decoder_model == "instantmesh_triplane":
                inputs, targets = batch["inputs"], batch["planes"]
            elif decoder_model == "nerf":
                inputs = batch["latent"]
                inputs = inputs.squeeze(1)
                targets = inputs.clone()  # Create a separate copy for targets
            else:
                inputs, targets = batch["inputs"], batch["outputs"]
                if decoder_model == "pc":
                    inputs = inputs.squeeze(2)
                    inputs = inputs.squeeze(2)
                    inputs, targets = targets, inputs

            if isinstance(model, (LTOrtho, LTpcOrtho, LTNeRFOrtho)):
                outputs, _ = model(inputs)
            else:
                outputs = model(inputs)

            # Ensure proper reshaping for targets based on outputs shape
            if decoder_model == "nerf":
                # For NeRF, ensure the targets are properly flattened to match outputs
                loss = criterion(outputs, targets.reshape(outputs.shape))
            else:
                # For other models, use the existing logic
                loss = criterion(outputs, targets.view(outputs.shape))

            total_loss += loss.item()

            if zero123_model:
                if i == 0 and epoch % 25 == 0 and plot_gifs:
                    with torch.no_grad():
                        multiview_outputs = generate_zero123_outputs(
                            zero123_model, outputs
                        )
                        multiview_gt = generate_zero123_outputs(zero123_model, targets)
                        wandb.log(
                            {
                                "train/multiview_outputs": wandb.Image(
                                    visualize_multiview_outputs(multiview_outputs)
                                ),
                                "train/multiview_gt": wandb.Image(
                                    visualize_multiview_outputs(multiview_gt)
                                ),
                            }
                        )
            elif decoder_model == "shape":
                if (
                    i == 0
                    and epoch % 25 == 0
                    and plot_gifs
                    or (use_decoder_supervision and i == 0)
                ):
                    with torch.no_grad():
                        prompt = [""]
                        with (
                            tempfile.NamedTemporaryFile(
                                suffix=".gif"
                            ) as ground_truth_gif_file,
                            tempfile.NamedTemporaryFile(
                                suffix=".gif"
                            ) as generated_gif_file,
                        ):

                            ground_truth_gif_path = ground_truth_gif_file.name
                            generated_gif_path = generated_gif_file.name
                            ground_truth_gif_filename = ground_truth_gif_path.replace(
                                ".gif", ""
                            )
                            generated_gif_filename = generated_gif_path.replace(
                                ".gif", ""
                            )

                            _ = decode(
                                shape_model,
                                targets[0].unsqueeze(0).half(),
                                prompt=prompt,
                                gif=True,
                                mesh=False,
                                export_to=[ground_truth_gif_filename],
                                return_trimesh=False,
                            ).images[0]
                            _ = decode(
                                shape_model,
                                outputs[0].unsqueeze(0).half(),
                                prompt=prompt,
                                gif=True,
                                mesh=False,
                                export_to=[generated_gif_filename],
                                return_trimesh=False,
                            ).images[0]

                            wandb.log(
                                {
                                    "val/ground_truth_gif": wandb.Video(
                                        ground_truth_gif_path, format="gif"
                                    ),
                                    "val/generated_gif": wandb.Video(
                                        generated_gif_path, format="gif"
                                    ),
                                }
                            )
            elif decoder_model == "lrm":
                if (i == 0 and epoch % 10 == 0 and plot_gifs) or (
                    use_decoder_supervision and i == 0
                ):
                    with (
                        tempfile.TemporaryDirectory() as gt_dir,
                        tempfile.TemporaryDirectory() as gen_dir,
                    ):
                        with torch.no_grad():
                            # print(targets[0].unsqueeze(0).shape, outputs[0].unsqueeze(0).shape)

                            ground_truth_gif_path, _ = lrm_decode(
                                targets[0].unsqueeze(0), lrm_model, True, True, gt_dir
                            )

                            try:
                                generated_gif_path, _ = lrm_decode(
                                    outputs[0].unsqueeze(0).view(1, 3, 48, 64, 64),
                                    lrm_model,
                                    True,
                                    True,
                                    gen_dir,
                                )
                            except IndexError:
                                generated_gif_path = None
                                print("Could not render gif")

                            # copy mp4 to current directory
                            # import shutil
                            # shutil.copy(ground_truth_gif_path, os.path.join(os.getcwd(), "ground_truth.mp4"))

                        if generated_gif_path:
                            wandb.log(
                                {
                                    "train/generated_gif": wandb.Video(
                                        generated_gif_path, format="mp4"
                                    ),
                                }
                            )
                        wandb.log(
                            {
                                "train/ground_truth_gif": wandb.Video(
                                    ground_truth_gif_path, format="mp4"
                                ),
                            }
                        )

    return total_loss / len(dataloader)


def get_optimizer(optimizer_name, model_parameters, lr):
    if optimizer_name.lower() == "adam":
        return optim.Adam(model_parameters, lr=lr)
    elif optimizer_name.lower() == "sgd":
        return optim.SGD(model_parameters, lr=lr)
    elif optimizer_name.lower() == "rmsprop":
        return optim.RMSprop(model_parameters, lr=lr)
    elif optimizer_name.lower() == "adamw":
        return optim.AdamW(model_parameters, lr=lr)
    elif optimizer_name.lower() == "muon":
        params_list = list(model_parameters)
        return Muon(
            muon_params=params_list,
            lr=lr,
            momentum=0.95,
            nesterov=True,
            ns_steps=6,
            adamw_params=None,
        )
    elif optimizer_name.lower() == "schedule_free":
        return AdamWScheduleFreeClosure(model_parameters, lr=lr)
    elif optimizer_name.lower() == "soap":
        return SOAP(model_parameters, lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


class WarmupCosineSchedule(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, t_total, cycles=0.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [
                base_lr * (self.last_epoch / self.warmup_steps)
                for base_lr in self.base_lrs
            ]
        else:
            progress = (self.last_epoch - self.warmup_steps) / (
                self.t_total - self.warmup_steps
            )
            return [
                base_lr
                * (0.5 * (1 + torch.cos(math.pi * self.cycles * 2.0 * progress)))
                for base_lr in self.base_lrs
            ]


def get_scheduler(scheduler_name, optimizer, scheduler_config):
    if scheduler_name.lower() == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_config)
    elif scheduler_name.lower() == "linear":

        def create_linear_decay_scheduler(
            optimizer,
            epochs_decay=500,
            steps_per_epoch=116,
            initial_lr=1e-2,
            final_lr=1e-6,
        ):
            def lr_lambda(step):
                current_epoch = step // steps_per_epoch
                if current_epoch >= epochs_decay:
                    return final_lr / initial_lr
                decay_steps = epochs_decay * steps_per_epoch
                current_step = step
                decay_rate = (final_lr - initial_lr) / decay_steps
                lr = initial_lr + decay_rate * current_step
                return lr / initial_lr

            return LambdaLR(optimizer, lr_lambda)

        return create_linear_decay_scheduler(optimizer, **scheduler_config)
    elif scheduler_name.lower() == "constant":
        return LambdaLR(optimizer, lr_lambda=lambda x: 1.0)
    elif scheduler_name.lower() == "warmup_cosine":
        return WarmupCosineSchedule(optimizer, **scheduler_config)
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def launch_training(
    seed=42,
    model_name="TwoLayerMLP",
    model_config={
        "input_shape": (257, 1024),
        "hidden_size": 512,
        "output_size": 1024 * 1024,
    },
    num_epochs=50,
    batch_size=32,
    learning_rate=0.001,
    project_dir="checkpoints/",
    wandb_project="streaming",
    run_name="initial_run",
    load_state=False,
    log_interval=10,
    gradient_accumulation_steps=4,
    clip_grad=True,
    max_grad_norm=1.0,
    use_profiler=False,
    optimizer_name="adam",
    scheduler_name="cosine",
    num_warmup_steps=0,
    dataset="rishitdagli/latents_1k",
    use_spectral_loss=False,
    load_shape=True,
    use_decoder_supervision=False,
    use_zero123=False,
    decoder_model="shape",
    plot_gifs=True,
    use_ortho_loss=False,
    scheduler_config={
        "epochs_decay": 500,
        "steps_per_epoch": 116,
        "initial_lr": 1e-2,
        "final_lr": 1e-6,
    },
):

    seed_all(seed)
    set_seed(seed)
    torch.backends.cudnn.benchmark = True

    dataloader_config = DataLoaderConfiguration(
        non_blocking=True,
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        log_with="wandb",
        project_config=ProjectConfiguration(
            project_dir=project_dir,
            automatic_checkpoint_naming=True,
            total_limit=2,
        ),
        dataloader_config=dataloader_config,
    )

    accelerator.init_trackers(
        project_name=wandb_project,
        init_kwargs={"wandb": {"name": run_name}},
        config={
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            **scheduler_config,
            "model_name": model_name,
            **model_config,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "clip_grad": clip_grad,
            "max_grad_norm": max_grad_norm,
            "optimizer": optimizer_name,
            "scheduler": scheduler_name,
            "num_warmup_steps": num_warmup_steps,
            "use_spectral_loss": use_spectral_loss,
            "decoder_model": decoder_model,
        },
    )

    try:
        dataset = load_dataset(dataset, split="train").with_format("torch")
    except ValueError as e:
        if "save_to_disk" in str(e):
            dataset = load_from_disk(dataset).with_format("torch")
        else:
            raise e
    train_dataset, val_dataset = dataset.train_test_split(0.1).values()

    train_dataloader, val_dataloader = create_dataloaders(
        train_dataset, val_dataset, batch_size, num_workers=12
    )

    if model_name == "TwoLayerMLP":
        model = TwoLayerMLP(**model_config)
    elif model_name == "ResidualNN":
        model = ResidualNN(**model_config)
    elif model_name == "TwoLayerCNN":
        model = TwoLayerCNN(**model_config)
    elif model_name == "ThreeLayerMLP":
        model = ThreeLayerMLP(**model_config)
    elif model_name == "ThreeLayerMLPWDropout":
        model = ThreeLayerMLP(**model_config, dropout=0.3)
    elif model_name == "ThreeLayerConvMLP":
        model = ThreeLayerConvMLP(**model_config)
    elif model_name == "LT":
        model = LT(**model_config)
    elif model_name == "LT2":
        model = LT(**model_config)
    elif model_name == "HighDimensionalUNet":
        model = HighDimensionalUNet(**model_config)
    elif model_name == "SimpleProjection":
        model = SimpleProjection(**model_config)
    elif model_name == "LT2Transformer":
        model = LT2Transformer(**model_config)
    elif model_name == "Const":
        model = ConstMLP(**model_config)
    elif model_name == "LTOrtho":
        model = LTOrtho(**model_config)
    elif model_name == "LTpc":
        model = LTpc(**model_config)
    elif model_name == "LTpcOrtho":
        model = LTpcOrtho(**model_config)
    elif model_name == "LTNeRF":
        model = LTNeRF(**model_config)
    elif model_name == "LTNeRFOrtho":
        model = LTNeRFOrtho(**model_config)

    # model.to(accelerator.device)
    criterion = nn.MSELoss()

    optimizer = get_optimizer(optimizer_name, model.parameters(), learning_rate)

    num_training_steps = num_epochs * len(train_dataloader)
    scheduler = get_scheduler(
        scheduler_name,
        optimizer,
        scheduler_config,
    )

    if optimizer_name == "muon":
        model, train_dataloader, val_dataloader = accelerator.prepare(
            model, train_dataloader, val_dataloader
        )
    else:
        model, optimizer, train_dataloader, val_dataloader, scheduler = (
            accelerator.prepare(
                model, optimizer, train_dataloader, val_dataloader, scheduler
            )
        )

    # if os.listdir(os.path.join(project_dir, "checkpoints")) and not load_state:
    #     raise ValueError("Checkpoints directory is not empty. Please set load_state=True")
    # if not os.path.exists(os.path.join(project_dir, "models")) and load_state:
    #     raise ValueError("Models directory does not exist. Please set load_state=False")

    accelerator.register_for_checkpointing(scheduler)
    if load_state:
        checkpoints = os.listdir(os.path.join(project_dir, "checkpoints"))
        latest_checkpoint = max(
            checkpoints, key=lambda x: int(re.search(r"\d+", x).group())
        )
        accelerator.load_state(
            os.path.join(project_dir, "checkpoints", latest_checkpoint)
        )
        accelerator.project_configuration.iteration = (
            int(re.search(r"\d+", latest_checkpoint).group()) + 1
        )

    shape_model = None
    zero123_model = None
    lrm_model = None
    if plot_gifs:
        if use_zero123:
            zero123_model = load_zero123_model(
                os.path.join(
                    os.path.dirname(__file__),
                    "../",
                    "../",
                    "weights",
                    "zero123",
                    "105000.ckpt",
                ),
                os.path.join(os.path.dirname(__file__), "zero123_conf.yaml"),
                device=accelerator.device,
            )
        else:
            if load_shape and decoder_model == "shape":
                shape_model = load_shape_pipeline(accelerator.device, fp16=True)
            elif decoder_model == "lrm":
                lrm_model = load_lrm(
                    "zxhezexin/openlrm-obj-base-1.1",
                    os.path.join(
                        os.path.dirname(__file__), "OpenLRM", "configs", "infer-b.yaml"
                    ),
                )

    # best_val_loss = float('inf')

    accelerator.save_state()

    for epoch in range(accelerator.project_configuration.iteration - 1, num_epochs + 1):
        train_loss = train_epoch(
            model,
            shape_model if load_shape else None,
            train_dataloader,
            optimizer,
            scheduler,
            criterion,
            accelerator,
            gradient_accumulation_steps,
            clip_grad,
            max_grad_norm,
            epoch,
            use_spectral_loss,
            use_decoder_supervision,
            "",
            True if plot_gifs else False,
            zero123_model if use_zero123 else None,
            lrm_model if decoder_model == "lrm" else None,
            decoder_model,
            use_ortho_loss,
        )
        val_loss = evaluate(
            model,
            shape_model if load_shape else None,
            val_dataloader,
            criterion,
            accelerator,
            "Validation",
            epoch,
            True if plot_gifs else False,
            use_decoder_supervision,
            zero123_model if use_zero123 else None,
            lrm_model if decoder_model == "lrm" else None,
            decoder_model,
            use_ortho_loss,
        )

        accelerator.log(
            {
                "val/loss": val_loss,
            }
        )

        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     checkpoint_path = os.path.join(project_dir, "models", f"{epoch}")
        #     accelerator.save_model(model, checkpoint_path, safe_serialization=True)
        # accelerator.log({"best_model_path": checkpoint_path})

        if (epoch + 1) % log_interval == 0:
            print(
                f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

        accelerator.save_state()

        scheduler.step()

    # if use_profiler:
    #     with accelerator.autocast(), torch.profiler.profile(
    #         activities=[
    #             torch.profiler.ProfilerActivity.CPU,
    #             torch.profiler.ProfilerActivity.CUDA,
    #         ],
    #         schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    #         on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiler_logs"),
    #         record_shapes=True,
    #         profile_memory=True,
    #         with_stack=True,
    #     ) as prof:
    #         for step, (inputs, targets) in enumerate(train_dataloader):
    #             if step >= 10:
    #                 break
    #             with accelerator.accumulate(model):
    #                 outputs = model(inputs)
    #                 loss = criterion(outputs, targets)
    #                 accelerator.backward(loss)
    #                 if accelerator.sync_gradients:
    #                     if clip_grad:
    #                         accelerator.clip_grad_norm_(
    #                             model.parameters(), max_grad_norm
    #                         )
    #                 optimizer.step()
    #                 scheduler.step()
    #                 optimizer.zero_grad(set_to_none=True)
    #             prof.step()

    #     print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    accelerator.end_training()


if __name__ == "__main__":
    launch_training()
