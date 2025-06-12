mkdir -p weights
cd weights

git clone https://huggingface.co/TencentARC/InstantMesh
mv InstantMesh instantmesh
cat > instantmesh/instant-mesh-large.yaml << 'EOL'
model_config:
  target: squeeze3d.instantmesh.models.lrm_mesh.InstantMesh
  params:
    encoder_feat_dim: 768
    encoder_freeze: false
    encoder_model_name: facebook/dino-vitb16
    transformer_dim: 1024
    transformer_layers: 16
    transformer_heads: 16
    triplane_low_res: 32
    triplane_high_res: 64
    triplane_dim: 80
    rendering_samples_per_ray: 128
    grid_res: 128
    grid_scale: 2.1


infer_config:
  unet_path: ${PWD}/instantmesh/diffusion_pytorch_model.bin
  model_path: ${PWD}/instantmesh/instant_mesh_large.ckpt
  texture_resolution: 1024
  render_resolution: 512
EOL

git clone https://huggingface.co/xiaohui2022/lion_ckpt
git clone https://huggingface.co/Yiwen-ntu/MeshAnything
git clone https://huggingface.co/mirshad7/NeRF-MAE
git clone https://huggingface.co/openai/shap-e

mkdir lrm
cd lrm
git clone https://huggingface.co/zxhezexin/openlrm-obj-base-1.1
cd ../..
