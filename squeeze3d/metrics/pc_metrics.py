import argparse
import os
import re
import subprocess
from chamferdist import ChamferDistance
import numpy as np

import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "../", "point-cloud-ssim"))
from registration_utils import read_point_cloud_from_ply_file, align
from ssim_utils import pc_ssim


def run_pc_error_and_extract_msef(file1, file2):
    command = [
        os.path.join(os.path.dirname(__file__), "../", "PCQM", "build", "pc_error"),
        "-a",
        file1,
        "-b",
        file2,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"Command failed with error: {result.stderr}")
    output = result.stdout
    match = re.search(r"mseF\s+\(p2point\):\s+(\d+(?:\.\d+)?)", output)

    if match:
        msef = float(match.group(1))
        return msef
    else:
        raise Exception("Could not find mseF value in the output")


def process_file(gt_dir, gen_dir, file, chamferDist, type):
    num_points = 10000

    pcA = read_point_cloud_from_ply_file(
        os.path.join(gt_dir, file),
        num_points=num_points,
        outlier_removal=False,
    )
    if type == "draco":
        pcB = read_point_cloud_from_ply_file(
            os.path.join(gen_dir, file[:-4] + "_decompressed.ply"),
            num_points=num_points,
            outlier_removal=False,
        )
    else:
        pcB = read_point_cloud_from_ply_file(
            os.path.join(gen_dir, file),
            num_points=num_points,
            outlier_removal=False,
        )
    pcA.estimate_normals()
    pcB.estimate_normals()
    pcB = align(pcB, pcA)

    pointssim = pc_ssim(
        pcA,
        pcB,
        neighborhood_size=12,
        feature="geometry",
        ref=0,
        estimators=["mean_ad"],
        pooling_methods=["mse"],
        const=np.finfo(float).eps,
    )[0][0]

    if type == "draco":
        pcqm = run_pc_error_and_extract_msef(
            os.path.join(gt_dir, file),
            os.path.join(gen_dir, file[:-4] + "_decompressed.ply"),
        )
    else:
        pcqm = run_pc_error_and_extract_msef(
            os.path.join(gt_dir, file), os.path.join(gen_dir, file)
        )

    points1 = np.asarray(pcA.points)
    points2 = np.asarray(pcB.points)
    points1_tensor = (
        torch.tensor(points1, dtype=torch.float32).unsqueeze(0).to("cuda")
    )  # [1, N, 3]
    points2_tensor = (
        torch.tensor(points2, dtype=torch.float32).unsqueeze(0).to("cuda")
    )  # [1, M, 3]

    chamfer = (
        chamferDist(points1_tensor, points2_tensor).detach().cpu().item()
        + chamferDist(points1_tensor, points2_tensor, reverse=True)
        .detach()
        .cpu()
        .item()
    )

    return {
        "pcqm": pcqm,
        "pointssim": pointssim,
        "chamfer": chamfer,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare rendered views of ground truth and generated meshes"
    )
    parser.add_argument(
        "--gt_dir", required=True, help="Directory containing ground truth meshes"
    )
    parser.add_argument(
        "--gen_dir", required=True, help="Directory containing generated meshes"
    )
    parser.add_argument(
        "--output_file", default="results.txt", help="File to save results to"
    )
    parser.add_argument("--type", default="all", help="Type of comparison to perform")
    args = parser.parse_args()

    files = []
    for item in os.listdir(args.gt_dir):
        if item.endswith(".ply"):
            files.append(item)

    if not files:
        print(f"No .ply files found in {args.gt_dir}")
        return

    all_metrics = {}
    chamferDist = ChamferDistance()

    for file in files:
        print(f"\nProcessing {file}...")
        metrics = process_file(args.gt_dir, args.gen_dir, file, chamferDist, args.type)
        all_metrics[file] = metrics

    # print the names of files with top 10 best pcqm
    print("\nTop 10 best pcqm:")
    sorted_metrics = sorted(all_metrics.items(), key=lambda x: x[1]["pcqm"])
    for i, (file, metrics) in enumerate(sorted_metrics[:10]):
        print(f"  {i+1}. {file}: {metrics['pcqm']:.4f}")

    if all_metrics:
        # Calculate mean values
        pcqm_values = [m["pcqm"] for m in all_metrics.values()]
        pointssim_values = [m["pointssim"] for m in all_metrics.values()]
        chamfer_values = [m["chamfer"] for m in all_metrics.values()]

        avg_pcqm = np.mean(pcqm_values)
        avg_pointssim = np.mean(pointssim_values)
        avg_chamfer = np.mean(chamfer_values)

        # Calculate standard deviations
        std_pcqm = np.std(pcqm_values)
        std_pointssim = np.std(pointssim_values)
        std_chamfer = np.std(chamfer_values)

        print("\nAverage metrics across all directories:")
        print(f"  Average PCQM: {avg_pcqm:.4f} (std: {std_pcqm:.4f})")
        print(f"  Average PointSSIM: {avg_pointssim:.4f} (std: {std_pointssim:.4f})")
        print(f"  Average Chamfer: {avg_chamfer:.4f} (std: {std_chamfer:.4f})")

        results_file = args.output_file

        with open(results_file, "w") as f:
            f.write("Mesh Comparison Results\n")
            f.write("======================\n\n")

            # for subdir, metrics in all_metrics.items():
            #     f.write(f"Metrics for {subdir}:\n")
            #     f.write(f"  PSNR: {metrics['psnr']:.4f}\n")
            #     f.write(f"  MS-SSIM: {metrics['ms_ssim']:.4f}\n")
            #     f.write(f"  LPIPS: {metrics['lpips']:.4f}\n\n")

            f.write("\nAverage metrics across all directories:\n")
            f.write(f"  Average PCQM: {avg_pcqm:.4f} (std: {std_pcqm:.4f})\n")
            f.write(
                f"  Average PointSSIM: {avg_pointssim:.4f} (std: {std_pointssim:.4f})\n"
            )
            f.write(f"  Average Chamfer: {avg_chamfer:.4f} (std: {std_chamfer:.4f})\n")

        print(f"Results saved to {results_file}")
    else:
        print("No valid results to report")


if __name__ == "__main__":
    main()
