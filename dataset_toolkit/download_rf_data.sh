mkdir -p data/NeRF-MAE/
cd data/
wget https://s3.amazonaws.com/tri-ml-public.s3.amazonaws.com/github/nerfmae/NeRF-MAE_pretrain.tar.gz
tar -xvf NeRF-MAE_pretrain.tar.gz
rm NeRF-MAE_pretrain.tar.gz
cd ..