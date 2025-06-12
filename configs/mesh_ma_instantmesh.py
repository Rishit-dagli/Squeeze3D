import os
import shutil

from squeeze3d.launch_training import launch_training

if os.path.exists("outputs/instantmesh/mesh_ma_instantmesh"):
    shutil.rmtree("outputs/instantmesh/mesh_ma_instantmesh")

launch_training(
    seed=3407,
    model_name="LTOrtho",
    model_config={
        "input_shape": (257, 1024),
        "hidden_size": 770,
        "output_size": 3 * 80 * 64 * 64,
        "dropout_prob": 0.35,
    },
    num_epochs=10000,
    batch_size=16,
    learning_rate=1e-3,
    project_dir=f"outputs/instantmesh/mesh_ma_instantmesh",
    wandb_project="squeeze3d",
    run_name=f"instantmesh/mesh_ma_instantmesh",
    load_state=False,
    log_interval=200,
    gradient_accumulation_steps=1,
    clip_grad=False,
    max_grad_norm=5.0,
    use_profiler=False,
    optimizer_name="muon",
    scheduler_name="linear",
    num_warmup_steps=0,
    dataset="rishitdagli/squeeze3d_mesh_instantmesh",
    use_spectral_loss=False,
    decoder_model="instantmesh_triplane",
    plot_gifs=False,
    scheduler_config={
        "epochs_decay": 600,
        "steps_per_epoch": 115,
        "initial_lr": 1e-3,
        "final_lr": 1e-7,
    },
    use_ortho_loss=True,
)
