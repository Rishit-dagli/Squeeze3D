import numpy as np
import requests
import tarfile

from registration_utils import read_point_cloud_from_ply_file, align
from ssim_utils import pc_ssim

NUM_POINTS = 10000

url = "http://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz"
target_path = "bunny.tar.gz"

response = requests.get(url, stream=True)
if response.status_code == 200:
    with open(target_path, "wb") as f:
        f.write(response.raw.read())

    my_tar = tarfile.open(target_path)
    my_tar.extractall(".")
    my_tar.close()

pcA = read_point_cloud_from_ply_file(
    "bunny/data/bun000.ply",
    num_points=NUM_POINTS,
    outlier_removal=False,
)

pcB = read_point_cloud_from_ply_file(
    "bunny/data/bun045.ply",
    num_points=NUM_POINTS,
    outlier_removal=False,
)

pcA.estimate_normals()
pcB.estimate_normals()

pcB = align(pcB, pcA)

score = pc_ssim(
    pcA,
    pcB,
    neighborhood_size=12,
    feature="curvature",
    ref=0,
    estimators=["std", "var", "mean_ad", "median_ad", "coef_var", "qcd"],
    pooling_methods=["mean", "mse", "rms"],
    const=np.finfo(float).eps,
)
print(score)
