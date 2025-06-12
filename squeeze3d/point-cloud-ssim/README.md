# point-cloud-ssim

This repository contains a python implementation of the point cloud structural similarity metric from the paper:

Alexiou, E., & Ebrahimi, T. (2020). Towards a Point Cloud Structural Similarity Metric. 2020 IEEE International Conference on Multimedia and Expo Workshops, ICMEW 2020, 1–6. https://doi.org/10.1109/ICMEW46912.2020.9106005

The implementation is based on the [Matlab implementation](https://github.com/mmspg/pointssim) disclosed in the same paper.

## Install environment

Assumes poetry is installed.

```
cd point-cloud-ssim
poetry install
```

## Calculate the structural similarity score between two point clouds

```bash
python main.py --pcA [filepath_stl_file_A] --pcB [filepath_stl_pcB]
```

For additional options please check the argument description in `main.py`.
This script performs alignment of the two point clouds using pcA as the target. 
