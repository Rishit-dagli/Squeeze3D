import numpy as np

from sklearn.neighbors import NearestNeighbors


def pc_ssim(
    pcA,
    pcB,
    neighborhood_size=12,
    feature="geometry",
    ref=0,
    estimators=["std", "var", "mean_ad", "median_ad", "coef_var", "qcd"],
    pooling_methods=["mean", "mse", "rms"],
    const=np.finfo(float).eps,
):
    """Structural similarity scores between point clouds A and B, which are
    represented by the corresponding custom structures pcA and pcB.

    Args:
        pcA (o3d.geometry.PointCloud): Point cloud structure.
        pcB (o3d.geometry.PointCloud): Point cloud structure.
        neighborhood_size (int): Number of nearest neighbors.
        feature (string): Feature to extract from the point cloud.
        ref (int): Symmetric or asymmetric structural similarity. 0, 1, or 2.
        estimators (list): List of statistical dispersion estimators.
        pooling_methods (list): List of pooling methods.
        const (float): Constant to avoid undefined operations.
    Returns:
        arr : Structural similarity scores. Shape is ESTIMATORSxPOOLING.
    """

    if feature not in ["geometry", "normal", "curvature"]:
        print("No analysis available for feature {}.".format(feature))
        return

    if ref not in [0, 1, 2]:
        print("Reference has to be either 0, 1, or 2!")
        return

    points_A = np.asarray(pcA.points)
    points_B = np.asarray(pcB.points)

    normals_A = np.asarray(pcA.normals)
    normals_B = np.asarray(pcB.normals)

    nbrs_A = NearestNeighbors(n_neighbors=neighborhood_size, algorithm="kd_tree").fit(
        points_A
    )

    nbrs_B = NearestNeighbors(n_neighbors=neighborhood_size, algorithm="kd_tree").fit(
        points_B
    )

    nbrs_BA = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(points_A)

    nbrs_AB = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(points_B)

    dist_A, ind_A = nbrs_A.kneighbors(points_A)
    dist_B, ind_B = nbrs_B.kneighbors(points_B)

    # CHECK
    dist_BA, ind_BA = nbrs_BA.kneighbors(points_B)
    dist_AB, ind_AB = nbrs_AB.kneighbors(points_A)

    if feature == "geometry":
        print("Structural similarity scores based on geometry related features.")
        geom_quant_A = dist_A[:, 1:]
        geom_quant_B = dist_B[:, 1:]

        ssim = ssim_score(
            geom_quant_A,
            geom_quant_B,
            ind_BA,
            ind_AB,
            ref=ref,
            estimators=estimators,
            pooling_methods=pooling_methods,
            const=const,
        )

    elif feature == "normal":
        print("Structural similarity scores based on normal-related features.")

        cos_sim_A = (
            np.sum(
                np.repeat(normals_A, neighborhood_size, axis=-1).reshape(
                    -1, neighborhood_size, 3
                )
                * normals_A[ind_A],
                axis=2,
            )
        ) / (
            np.linalg.norm(
                np.repeat(normals_A, neighborhood_size, axis=-1).reshape(
                    -1, neighborhood_size, 3
                ),
                axis=2,
            )
            * np.linalg.norm(normals_A[ind_A], axis=2)
        )

        cos_sim_B = (
            np.sum(
                np.repeat(normals_B, neighborhood_size, axis=-1).reshape(
                    -1, neighborhood_size, 3
                )
                * normals_B[ind_B],
                axis=2,
            )
        ) / (
            np.linalg.norm(
                np.repeat(normals_A, neighborhood_size, axis=-1).reshape(
                    -1, neighborhood_size, 3
                ),
                axis=2,
            )
            * np.linalg.norm(normals_B[ind_B], axis=2)
        )

        norm_quant_A = cos_sim_A[:, 1:]
        norm_quant_B = cos_sim_B[:, 1:]

        ssim = ssim_score(
            norm_quant_A,
            norm_quant_B,
            ind_BA,
            ind_AB,
            ref=ref,
            estimators=estimators,
            pooling_methods=pooling_methods,
            const=const,
        )

    elif feature == "curvature":
        print("Structural similarity scores based on curvature features.")
        curv_quant_A = estimate_curvature(points_A, ind_A)
        curv_quant_B = estimate_curvature(points_B, ind_B)

        ssim = ssim_score(
            curv_quant_A,
            curv_quant_B[:, 1:],
            ind_BA,
            ind_AB,
            ref=ref,
            estimators=estimators,
            pooling_methods=pooling_methods,
            const=const,
        )

    return ssim


def ssim_score(
    quant_A,
    quant_B,
    ind_BA,
    ind_AB,
    ref=0,
    estimators=["std", "var", "mean_ad", "median_ad", "coef_var", "qcd"],
    pooling_methods=["mean", "mse", "rms"],
    const=np.finfo(float).eps,
):
    """Returns structural similarity score based on specific features.

    Args:
        quant_A (arr): Array with features. NUM_POINTSxNEIGHBORS.
        quant_B (arr): Array with features. NUM_POINTSxNEIGHBORS.
        ref (int): Symmetric or asymmetric structural similarity. 0, 1, or 2.
        estimators (list): List of statistical dispersion estimators.
        pooling_methods (list): List of pooling methods.
        const (float): Constant to avoid undefined operations.
    Returns:
        arr : Structural similarity scores. Shape is ESTIMATORSxPOOLING.
    """

    # Feature map extraction
    f_map_A = feature_map(quant_A, estimators=estimators)

    f_map_B = feature_map(quant_B, estimators=estimators)

    # Structural similarity of score B (set A as a reference)
    if ref == 0 or ref == 1:
        ssim_BA = np.zeros((len(estimators), len(pooling_methods)))
        for i, estimator in enumerate(estimators):
            error_map_BA = error_map(f_map_B[:, i], f_map_A[:, i], ind_BA, const)
            ssim_map_BA = 1 - error_map_BA

            ssim_BA[i, :] = pooling(ssim_map_BA, pooling_methods)

    # Structural similarity score of A (set B as a reference)
    if ref == 0 or ref == 2:
        ssim_AB = np.zeros((len(estimators), len(pooling_methods)))
        for i, estimator in enumerate(estimators):
            error_map_AB = error_map(f_map_A[:, i], f_map_B[:, i], ind_AB, const)
            ssim_map_AB = 1 - error_map_AB

            ssim_AB[i, :] = pooling(ssim_map_AB, pooling_methods)
    # Symmetric structural similarity score
    if ref == 0:
        ssim = np.minimum(ssim_BA, ssim_AB)
        print("Symmetric Structural Similarity.")
    elif ref == 1:
        ssim = ssim_BA
        print("Structural similarity of score B (set A as a reference).")
    else:
        ssim = ssim_AB
        print("Structural similarity score of A (set B as a reference).")

    return ssim


def feature_map(
    quant, estimators=["std", "var", "mean_ad", "median_ad", "coef_var", "qcd"]
):
    """Returns the feature map of a point cloud based
    on the feature quantities and the estimators for
    statistical dispersion.

    Args:
        quant (arr): Per feature quantities. NUM_POINTSxNEIGHBORS.
        estimators (list): List of statistical dispersion estimators.
    Returns:
        arr: Feature map per estimator. NUM_POINTSxESTIMATORS.
    """

    f_map = np.zeros((quant.shape[0], len(estimators)))

    for k, estimator in enumerate(estimators):
        if estimator == "std":
            f_map[:, k] = np.std(quant, axis=1)
        elif estimator == "var":
            f_map[:, k] = np.var(quant, axis=1)
        elif estimator == "mean_ad":
            f_map[:, k] = np.mean(
                np.abs(quant - np.reshape(np.mean(quant, axis=1), (-1, 1))),
                axis=1,
            )
        elif estimator == "median_ad":
            f_map[:, k] = np.median(
                np.abs(quant - np.reshape(np.median(quant, axis=1), (-1, 1))),
                axis=1,
            )
        elif estimator == "coef_var":
            f_map[:, k] = np.std(quant, axis=1) / np.mean(quant, axis=1)
        elif estimator == "qcd":
            qq = np.quantile(quant, [0.25, 0.75], axis=1).T
            f_map[:, k] = (qq[:, 1] - qq[:, 0]) / (qq[:, 1] + qq[:, 0])
        else:
            print("Wrong input!")

    return f_map


def error_map(f_map_Y, f_map_X, ind_YX, const):
    """Returns error map of point cloud Y, based on the the relative difference
    between feature maps of X and Y.

    Args:
        f_map_Y (arr): Feature map of point cloud X.
        f_map_X ([type]): [description]
        ind_YX ([type]): [description]
        const ([type]): [description]

    Returns:
        [type]: [description]
    """

    error_map_YX = np.abs(f_map_X[ind_YX] - np.reshape(f_map_Y, (-1, 1))) / (
        np.amax(
            np.maximum(np.abs(f_map_X[ind_YX]), np.abs(np.reshape(f_map_Y, (-1, 1)))),
            axis=1,
        ).reshape(-1, 1)
        + const
    )

    return error_map_YX


def estimate_curvature(pc_points, points_x_neighbors):
    """Return curvature estimations based on
    their respective neighborhoods.

    Args:
        pc_points (arr): Points in the point cloud. NUM_POINTSx3.
        points_x_neighbors (arr): Points and indices of neighbors.
                                NUM_POINTSxNEIGHBORS.

    Returns:
        arr: Curvature values of the neighborhood for each point.
            NUM_POINTSxNEIGHBORS.
    """

    curvature = np.zeros(len(pc_points))
    neighbor_point_coordinates = pc_points[points_x_neighbors]
    curv_features = np.zeros(points_x_neighbors.shape)

    for i in range(len(pc_points)):
        M = neighbor_point_coordinates[i]
        M = M.T
        M = np.cov(M)
        V, _ = np.linalg.eig(M)
        h1, h2, h3 = V
        curvature[i] = h3 / (h1 + h2 + h3)

    for k in range(len(points_x_neighbors)):
        curv_features[k, :] = curvature[points_x_neighbors[k]]

    return curv_features


def pooling(q_map, pooling_methods):
    """Score of a point cloud based on different pooling methods.

    Args:
        q_map (arr): Quality map of a point cloud.
        pooling_methods (list): List of pooling methods.

    Returns:
        score: Quality score of a point cloud, per pooling method.
                1xPOOLING.
    """

    score = np.zeros(len(pooling_methods))

    for i, method in enumerate(pooling_methods):
        if method == "mean":
            score[i] = np.nanmean(q_map)
        elif method == "mse":
            score[i] = np.nanmean(q_map**2)
        elif method == "rms":
            score[i] = np.sqrt(np.nanmean(q_map**2))
        else:
            print("Wrong input!")
    return score
