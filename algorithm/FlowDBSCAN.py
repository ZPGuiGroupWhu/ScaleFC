"""
FlowDBSCAN, implemented by scikit-learn

Paper reference: Tao R. and Thill J.-C. (2016). “A Density-Based Spatial Flow Cluster Detection Method.” International Conference on GIScience Short Paper Proceedings 1.

How to use:
    from algorithm.FlowDBSCAN import flow_cluster_dbscan
    OD = ...
    label = flow_cluster_dbscan(OD, eps=0.5, min_flows=5, n_jobs=-1)
"""

from algorithm.util_tools import flow_distance_matrix_max_euclidean, flow_check_OD
from sklearn.cluster import dbscan


def flow_cluster_DBSCAN(OD, eps, min_flows, n_jobs=None):
    flow_check_OD(OD)
    _, labels = dbscan(X=flow_distance_matrix_max_euclidean(
        OD), eps=eps, min_samples=min_flows, metric="precomputed", n_jobs=n_jobs)
    return labels
