import torch
import trimesh
from diffusers import ShapEPipeline
from diffusers.utils import export_to_gif


def load_shape_pipeline(device, ckpt_path="openai/shap-e", fp16=False):
    if fp16:
        model = ShapEPipeline.from_pretrained(
            ckpt_path, torch_dtype=torch.float16, variant="fp16"
        )
    else:
        model = ShapEPipeline.from_pretrained(ckpt_path)
    model = model.to(device)
    return model


def make_latents(model, prompt=["A firecracker", "A birthday cupcake"]):
    latents = model(
        prompt, num_inference_steps=64, guidance_scale=4.0, output_type="latent"
    ).images
    latents = latents.view(latents.shape[0], -1)
    return latents


def decode(
    model,
    latents,
    prompt=["A firecracker", "A birthday cupcake"],
    gif=False,
    mesh=True,
    export_to=["firecracker_3d", "cake_3d"],
    return_trimesh=True,
    num_inference_steps=64,
):
    outputs = model(
        prompt,
        latents=latents,
        num_inference_steps=num_inference_steps,
        guidance_scale=4.0,
        frame_size=256,
        output_type="mesh" if mesh else "pil",
    )

    if export_to:
        for i, output in enumerate(outputs.images):
            if gif and not mesh:
                export_to_gif(output, f"{export_to[i]}.gif")

            if mesh:
                obj_filename = f"{export_to[i]}.obj"
                vertices = output.verts.detach().cpu().numpy()
                faces = output.faces.detach().cpu().numpy()
                mesh_obj = trimesh.Trimesh(vertices=vertices, faces=faces)
                mesh_obj.export(obj_filename)
                if return_trimesh:
                    return mesh_obj

    return outputs


# model = load_shape_pipeline("cuda")
# latents = make_latents(model)
# results = decode(model, latents)

# model = load_shape_pipeline("cuda")
# print(make_latents(model).shape)
