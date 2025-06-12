import os
import shutil

from squeeze3d.launch_training import launch_training

if os.path.exists("outputs/shape/mesh_ma_shape"):
    shutil.rmtree("outputs/shape/mesh_ma_shape")

launch_training(
    seed=3407,
    model_name="LTOrtho",
    model_config={
        "input_shape": (257, 1024),
        "hidden_size": 752,
        "output_size": 1024 * 1024,
        "dropout_prob": 0.35,
    },
    num_epochs=10000,
    batch_size=8,
    learning_rate=1e-2,
    project_dir=f"outputs/shape/mesh_ma_shape",
    wandb_project="squeeze3d",
    run_name=f"shape/mesh_ma_shape",
    load_state=False,
    log_interval=200,
    gradient_accumulation_steps=1,
    clip_grad=False,
    max_grad_norm=5.0,
    use_profiler=False,
    optimizer_name="adam",
    scheduler_name="linear",
    num_warmup_steps=0,
    dataset="rishitdagli/squeeze3d_mesh_shape",
    use_spectral_loss=False,
    decoder_model="shape",
    plot_gifs=False,
    scheduler_config={
        "epochs_decay": 700,
        "steps_per_epoch": 113,
        "initial_lr": 1e-4,
        "final_lr": 1e-4,
    },
    use_ortho_loss=True,
)
