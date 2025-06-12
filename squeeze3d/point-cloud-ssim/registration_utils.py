# The following registration code is based on the official Open3D tutorials

import copy
import numpy as np
import open3d as o3d


def align(source, target):
    # Perform global registration
    voxel_size = 0.05  # means 5cm for this dataset
    (
        source,
        target,
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
    ) = prepare_dataset(source, target, voxel_size)

    result_ransac = execute_global_registration(
        source_down, target_down, source_fpfh, target_fpfh, voxel_size
    )

    # Local refinement
    reg_p2p = refine_registration(
        source, target, result_ransac.transformation, voxel_size
    )

    source_transformed = copy.deepcopy(source)
    source_transformed.transform(reg_p2p.transformation)

    return source_transformed


def refine_registration(source, target, init_transformation, voxel_size):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source,
        target,
        distance_threshold,
        init_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    return reg_p2p


def execute_global_registration(
    source_down, target_down, source_fpfh, target_fpfh, voxel_size
):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold
            ),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )
    return result


def prepare_dataset(source, target, voxel_size):
    print("Disturb initial pose.")
    trans_init = np.asarray(
        [
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    source.transform(trans_init)

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )
    return pcd_down, pcd_fpfh


def read_point_cloud_from_stl_file(filepath, num_points=1024, outlier_removal=True):
    mesh = o3d.io.read_triangle_mesh(filepath)
    pc = mesh.sample_points_uniformly(number_of_points=num_points)
    if outlier_removal:
        pc, _ = pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return pc


def read_point_cloud_from_ply_file(filepath, num_points=1024, outlier_removal=True):
    pc = o3d.io.read_point_cloud(filepath)
    if outlier_removal:
        pc, _ = pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return pc
