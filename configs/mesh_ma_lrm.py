import os
import shutil

from squeeze3d.launch_training import launch_training

if os.path.exists("outputs/lrm/mesh_ma_lrm"):
    shutil.rmtree("outputs/lrm/mesh_ma_lrm")

launch_training(
    seed=3407,
    model_name="LTOrtho",
    model_config={
        "input_shape": (257, 1024),
        "hidden_size": 1024,
        "output_size": 3 * 48 * 64 * 64,
        "dropout_prob": 0.35,
    },
    num_epochs=10000,
    batch_size=16,
    learning_rate=1e-2,
    project_dir=f"outputs/lrm/mesh_ma_lrm",
    wandb_project="squeeze3d",
    run_name=f"lrm/mesh_ma_lrm",
    load_state=False,
    log_interval=200,
    gradient_accumulation_steps=1,
    clip_grad=False,
    max_grad_norm=5.0,
    use_profiler=False,
    optimizer_name="muon",
    scheduler_name="linear",
    num_warmup_steps=0,
    dataset="rishitdagli/squeeze3d_mesh_lrm",
    use_spectral_loss=False,
    decoder_model="lrm",
    plot_gifs=False,
    scheduler_config={
        "epochs_decay": 700,
        "steps_per_epoch": 113,
        "initial_lr": 1e-2,
        "final_lr": 1e-7,
    },
    use_ortho_loss=True,
)
