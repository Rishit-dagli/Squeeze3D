import os
import shutil

from squeeze3d.launch_training import launch_training

if os.path.exists("outputs/pc/pc_pn_lion_8192"):
    shutil.rmtree("outputs/pc/pc_pn_lion_8192")

launch_training(
    seed=3407,
    model_name="LTpcOrtho",
    model_config={
        "hidden_size": 8192,
        "dropout_prob": 0.3,
    },
    num_epochs=10000,
    batch_size=16,
    learning_rate=1e-3,
    project_dir=f"outputs/pc/pc_pn_lion_8192",
    wandb_project="squeeze3d",
    run_name=f"pc/pc_pn_lion_8192",
    load_state=False,
    log_interval=200,
    gradient_accumulation_steps=1,
    clip_grad=False,
    max_grad_norm=5.0,
    use_profiler=False,
    optimizer_name="muon",
    scheduler_name="linear",
    num_warmup_steps=0,
    dataset="rishitdagli/squeeze3d_pc_lion",
    use_spectral_loss=False,
    decoder_model="pc",
    plot_gifs=False,
    scheduler_config={
        "epochs_decay": 1000,
        "steps_per_epoch": 113,
        "initial_lr": 1e-3,
        "final_lr": 1e-7,
    },
    use_ortho_loss=True,
)
