import os
import shutil

from squeeze3d.launch_training import launch_training

if os.path.exists("outputs/nerf/rf_nerfmae"):
    shutil.rmtree("outputs/nerf/rf_nerfmae")

launch_training(
    seed=3407,
    model_name="LTNeRFOrtho",
    model_config={
        "dropout": 0.2,
    },
    num_epochs=10000,
    batch_size=4,
    learning_rate=1e-2,
    project_dir=f"outputs/nerf/rf_nerfmae",
    wandb_project="squeeze3d",
    run_name=f"nerf/rf_nerfmae",
    load_state=False,
    log_interval=200,
    gradient_accumulation_steps=1,
    clip_grad=False,
    max_grad_norm=5.0,
    use_profiler=False,
    optimizer_name="muon",
    scheduler_name="linear",
    num_warmup_steps=0,
    dataset="rishitdagli/squeeze3d_rf_nerfmae",
    use_spectral_loss=False,
    decoder_model="nerf",
    plot_gifs=False,
    scheduler_config={
        "epochs_decay": 1000,
        "steps_per_epoch": 200,
        "initial_lr": 1e-2,
        "final_lr": 1e-5,
    },
    use_ortho_loss=True,
)
