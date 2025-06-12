import os
import subprocess
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from squeeze3d.OpenLRM.openlrm.runners import REGISTRY_RUNNERS
from squeeze3d.OpenLRM.openlrm.runners.infer.base_inferrer import Inferrer
from squeeze3d.OpenLRM.openlrm.utils.preprocess import Preprocessor


def load_lrm(model_name: str, config_path: str):
    os.environ.update(
        {
            "APP_ENABLED": "1",
            "APP_MODEL_NAME": model_name,
            "APP_INFER": config_path,
            "APP_TYPE": "infer.lrm",
            "NUMBA_THREADING_LAYER": "omp",
        }
    )

    # preprocessor = Preprocessor()
    InferrerClass: Inferrer = REGISTRY_RUNNERS[os.getenv("APP_TYPE")]
    inferrer = InferrerClass()
    inferrer.__enter__()

    return inferrer


def infer_lrm(inferrer, image_path: str, output_dir: str, source_cam_dist: float = 2.0):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    temp_video_path = output_dir / "temp_output.mp4"
    final_video_path = output_dir / "output.mp4"
    mesh_path = output_dir / "output.ply"

    inferrer.infer_single(
        image_path=str(image_path),
        source_cam_dist=source_cam_dist,
        export_video=True,
        export_mesh=True,
        dump_video_path=str(temp_video_path),
        dump_mesh_path=str(mesh_path),
    )

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(temp_video_path),
        "-c:v",
        "libx264",
        "-preset",
        "veryslow",
        "-crf",
        "17",
        "-pix_fmt",
        "yuv420p",
        "-vf",
        "fps=30",
        str(final_video_path),
    ]

    try:
        subprocess.run(command, check=True, capture_output=True)
        os.remove(temp_video_path)
    except subprocess.CalledProcessError as e:
        print(f"Error encoding video: {e.stderr.decode()}")
        final_video_path = temp_video_path

    return str(final_video_path), str(mesh_path)


def lrm_encode(input_path: str, model):
    input_path = Image.open(input_path)
    return model.encode(input_path)


def lrm_decode(latents, model, export_video: bool, export_mesh: bool, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    temp_video_path = output_dir / "temp_output.mp4"
    final_video_path = output_dir / "output.mp4"
    mesh_path = output_dir / "output.obj"

    model.decode(
        planes=latents,
        export_video=export_video,
        export_mesh=export_mesh,
        dump_video_path=str(temp_video_path),
        dump_mesh_path=str(mesh_path),
    )

    if export_video:
        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(temp_video_path),
            "-c:v",
            "libx264",
            "-preset",
            "veryslow",
            "-crf",
            "17",
            "-pix_fmt",
            "yuv420p",
            "-vf",
            "fps=30",
            str(final_video_path),
        ]
        subprocess.run(command, check=True, capture_output=True)
        os.remove(temp_video_path)

    return os.path.abspath(final_video_path), os.path.abspath(mesh_path)
