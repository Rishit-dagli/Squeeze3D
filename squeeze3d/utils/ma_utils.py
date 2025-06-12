import argparse
import os

import mesh2sdf.core
import numpy as np
import skimage.measure
import torch
import torchvision
import trimesh
from einops import rearrange
from omegaconf import OmegaConf
from safetensors import safe_open
from safetensors.torch import safe_open

from ..meshanything2.meshanything_v2 import MeshAnythingV2

from ..meshanything.meshanything import MeshAnything


def normalize_vertices(vertices, scale=0.9):
    bbmin, bbmax = vertices.min(0), vertices.max(0)
    center = (bbmin + bbmax) * 0.5
    scale = 2.0 * scale / (bbmax - bbmin).max()
    vertices = (vertices - center) * scale
    return vertices, center, scale


def export_to_watertight(normalized_mesh, octree_depth: int = 7):
    size = 2**octree_depth
    level = 2 / size

    scaled_vertices, to_orig_center, to_orig_scale = normalize_vertices(
        normalized_mesh.vertices
    )

    sdf = mesh2sdf.core.compute(scaled_vertices, normalized_mesh.faces, size=size)

    vertices, faces, normals, _ = skimage.measure.marching_cubes(np.abs(sdf), level)

    # watertight mesh
    vertices = vertices / size * 2 - 1  # -1 to 1
    vertices = vertices / to_orig_scale + to_orig_center
    # vertices = vertices / to_orig_scale + to_orig_center
    mesh = trimesh.Trimesh(vertices, faces, normals=normals)

    return mesh


def process_mesh_to_pc(mesh_list, marching_cubes=False, sample_num=4096):
    # mesh_list : list of trimesh
    pc_normal_list = []
    return_mesh_list = []
    for mesh in mesh_list:
        if marching_cubes:
            mesh = export_to_watertight(mesh)
            print("MC over!")
        return_mesh_list.append(mesh)
        points, face_idx = mesh.sample(sample_num, return_index=True)
        normals = mesh.face_normals[face_idx]

        pc_normal = np.concatenate([points, normals], axis=-1, dtype=np.float16)
        pc_normal_list.append(pc_normal)
        print("process mesh success")
    return pc_normal_list, return_mesh_list


def get_args():
    parser = argparse.ArgumentParser("MeshAnything", add_help=False)
    parser.add_argument("--llm", default="facebook/opt-350m", type=str)
    parser.add_argument("--codebook_size", default=8192, type=int)
    parser.add_argument("--codebook_dim", default=1024, type=int)
    parser.add_argument("--n_max_triangles", default=800, type=int)
    args, _ = parser.parse_known_args()  # Ignore unrecognized arguments
    return args


def load_meshanything(
    args,
    ckpt_path=os.path.join(
        os.path.dirname(__file__),
        "../",
        "../",
        "weights",
        "MeshAnything",
        "MeshAnything_350m.pth",
    ),
    device="cuda:0",
):
    model = MeshAnything(args)
    tensors = {}

    with safe_open(ckpt_path, framework="pt", device=device) as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    model.load_state_dict(tensors, strict=True)

    print(f"Successfully loaded {len(tensors)} tensors from checkpoint")

    point_encoder_keys = [k for k in tensors.keys() if k.startswith("point_encoder")]
    if point_encoder_keys:
        print(f"Found {len(point_encoder_keys)} keys related to point_encoder")
    else:
        print("Warning: No point_encoder weights found in the checkpoint")
    model = model.to(device)
    return model


def load_meshanything2():
    model = MeshAnythingV2.from_pretrained("Yiwen-ntu/meshanythingv2")
    return model


def process_mesh(mesh_path):
    mesh = trimesh.load(mesh_path)
    vertices = mesh.vertices
    bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])

    vertices = vertices - (bounds[0] + bounds[1])[None, :] / 2
    vertices = vertices / (bounds[1] - bounds[0]).max()
    mesh.vertices = vertices

    mesh.merge_vertices()
    mesh.update_faces(mesh.unique_faces())
    mesh.fix_normals()

    points, face_indices = mesh.sample(4096, return_index=True)
    normals = mesh.face_normals[face_indices]

    points = points / np.abs(points).max() * 0.9995

    pc_normal = np.concatenate([points, normals], axis=-1).astype(np.float32)
    return pc_normal


def main(input_path):
    device = "cuda:0"
    args = get_args()
    model = load_meshanything(
        args,
        ckpt_path=os.path.join(
            os.path.dirname(__file__),
            "../",
            "../",
            "weights",
            "MeshAnything",
            "MeshAnything_350m.pth",
        ),
        device=device,
    )
    model.eval()

    pc_normal = process_mesh(input_path)

    pc_normal = torch.tensor(pc_normal, dtype=torch.float32).unsqueeze(0).to(device)

    latents = []
    with torch.no_grad():
        for _ in range(2):
            point_feature = model.point_encoder.encode_latents(pc_normal)
            processed_point_feature = model.process_point_feature(point_feature)
            latents.append(processed_point_feature)

    diff = torch.abs(latents[0] - latents[1])
    max_diff = torch.max(diff).item()
    mean_diff = torch.mean(diff).item()

    print("Latent shape:", latents[0].shape)
    print("Maximum difference between latents:", max_diff)
    print("Mean difference between latents:", mean_diff)
    print("First few values of first latent:", latents[0][0, 0, :10])
    print("First few values of second latent:", latents[1][0, 0, :10])
