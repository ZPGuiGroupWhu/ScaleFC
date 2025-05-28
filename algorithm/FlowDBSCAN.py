"""
FlowDBSCAN, implemented by scikit-learn

Paper reference: Tao R. and Thill J.-C. (2016). “A Density-Based Spatial Flow Cluster Detection Method.” International Conference on GIScience Short Paper Proceedings 1.

How to use:
    from algorithm.FlowDBSCAN import flow_cluster_dbscan
    OD = ...
    label = flow_cluster_dbscan(OD, eps=0.5, min_flows=5, n_jobs=-1)
"""

from typing import Callable
from joblib import Parallel, delayed
from algorithm.util_tools import flow_distance_matrix_max_euclidean, flow_check_OD
from sklearn.cluster import dbscan
from algorithm.ScaleFC import _scale_factor_func_linear
import numpy as np


def flow_cluster_DBSCAN(OD, eps, min_flows, n_jobs=None, flow_distance_func: Callable = flow_distance_matrix_max_euclidean):
    flow_check_OD(OD)
    _, labels = dbscan(X=flow_distance_func(
        OD), eps=eps, min_samples=min_flows, metric="precomputed", n_jobs=n_jobs)
    return labels


def flow_cluster_DBSCAN_with_scale_factor(OD, scale_factor, min_flows, n_jobs=None, flow_distance_func: Callable = flow_distance_matrix_max_euclidean):
    # First find core flows
    distance_matrix = flow_distance_func(OD)
    eps_array = _scale_factor_func_linear(OD, scale_factor)
    minpts = min_flows

    n, m = distance_matrix.shape
    assert n == m, "distance_matrix must be a square matrix."
    assert n == len(eps_array), "distance_matrix.shape[0] must be equal to len(eps_array)."

    labels = np.full(n, -1)  # Initialize all point labels as -1, indicating unclassified

    def region_query(point_idx):
        return np.where(distance_matrix[point_idx] <= eps_array[point_idx])[0]

    def expand_cluster(point_idx, neighbors, cluster_id):
        cluster = set([point_idx])
        queue = list(neighbors)
        while queue:
            q = queue.pop(0)
            # Unclassified
            if labels[q] == -1:  # Unclassified
                labels[q] = cluster_id
                cluster.add(q)
                new_neighbors = region_query(q)
                if len(new_neighbors) >= minpts:
                    queue.extend([x for x in new_neighbors if x not in cluster])
            # Noise point
            elif labels[q] == -2:  # Noise point
                labels[q] = cluster_id
                cluster.add(q)
        return list(cluster)

    def process_point(point_idx):
        # Skip if already classified
        if labels[point_idx] != -1:
            return None 
        neighbors = region_query(point_idx)
        if len(neighbors) < minpts:
            # Mark as noise
            labels[point_idx] = -2  # Mark as noise
            return None
        else:
            return expand_cluster(point_idx, neighbors, point_idx)

    # Process all points in parallel
    clusters = Parallel(n_jobs=n_jobs)(delayed(process_point)(i) for i in range(n))

    # Relabel clusters
    valid_clusters = [c for c in clusters if c is not None]
    for i, cluster in enumerate(valid_clusters):
        for point in cluster:
            labels[point] = i

    def clusters_relabel_by_order(cluster_label: np.ndarray):
        cluster_label = np.asarray(cluster_label, dtype=int)
        new_label = np.full(len(cluster_label), -1)
        _, idx = np.unique(cluster_label, return_index=True)

        sorted_idx = np.sort(idx)
        i = 0
        for x in sorted_idx:
            cur_l = cluster_label[x]
            if cur_l == -1:
                continue
            new_label[cluster_label == cur_l] = i
            i += 1
        return new_label

    return clusters_relabel_by_order(labels)