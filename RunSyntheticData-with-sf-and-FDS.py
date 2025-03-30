
import sys

sys.path.append(".")

from algorithm.FlowLF import flow_cluster_LF, flow_cluster_LF_with_scale_factor
from algorithm.ScaleFC import flow_cluster_ScaleFC
from algorithm.FlowDBSCAN import flow_cluster_DBSCAN, flow_cluster_DBSCAN_with_scale_factor
from algorithm.util_tools import *
from RunSyntheticData import ClusterExternalIndexAny, ODAlgorithmInfo

import pandas as pd
import numpy as np


all_data_algo_infos = {
    "A": [
        ODAlgorithmInfo(name="FlowLF", func=flow_cluster_LF,
                        other_func_kwargs=dict(lambda_=1, radius=3, n_jobs=-1)),
        ODAlgorithmInfo(name="FlowDBSCAN", func=flow_cluster_DBSCAN,
                        other_func_kwargs=dict(eps=2.5, min_flows=5, n_jobs=-1)),
        ODAlgorithmInfo(name="FlowLF (SF)", func=flow_cluster_LF_with_scale_factor,
                        other_func_kwargs=dict(scale_factor=0.26, n_jobs=-1)),
        ODAlgorithmInfo(name="FlowDBSCAN (SF)", func=flow_cluster_DBSCAN_with_scale_factor,
                        other_func_kwargs=dict(scale_factor=0.2, min_flows=5, n_jobs=-1)),
        ODAlgorithmInfo(name="FlowLF (FDS)", func=flow_cluster_LF,
                        other_func_kwargs=dict(lambda_=1, radius=0.15, n_jobs=-1, flow_distance_func=flow_distance_matrix_weighted_with_length, flow_distance_other_flow_func=flow_distance_other_flow_matrix_weighted_with_length)),
        ODAlgorithmInfo(name="FlowDBSCAN (FDS)", func=flow_cluster_DBSCAN,
                        other_func_kwargs=dict(eps=0.1, min_flows=5, n_jobs=-1, flow_distance_func=flow_distance_matrix_weighted_with_length)),
        ODAlgorithmInfo(name="ScaleFC", func=flow_cluster_ScaleFC, other_func_kwargs=dict(
            scale_factor=0.2, min_flows=4, n_jobs=-1, can_discard_flow_group_func=lambda x, y: len(x) < 8))
    ],

    "B": [
        ODAlgorithmInfo(name="FlowLF", func=flow_cluster_LF,
                        other_func_kwargs=dict(lambda_=1, radius=2, n_jobs=-1)),
        ODAlgorithmInfo(name="FlowDBSCAN", func=flow_cluster_DBSCAN,
                        other_func_kwargs=dict(eps=1.6, min_flows=5, n_jobs=-1)),
        ODAlgorithmInfo(name="FlowLF (SF)", func=flow_cluster_LF_with_scale_factor,
                        other_func_kwargs=dict(scale_factor=0.23, n_jobs=-1)),
        ODAlgorithmInfo(name="FlowDBSCAN (SF)", func=flow_cluster_DBSCAN_with_scale_factor,
                        other_func_kwargs=dict(scale_factor=0.18, min_flows=5, n_jobs=-1)),
        ODAlgorithmInfo(name="FlowLF (FDS)", func=flow_cluster_LF,
                        other_func_kwargs=dict(lambda_=1, radius=0.13, n_jobs=-1, flow_distance_func=flow_distance_matrix_weighted_with_length, flow_distance_other_flow_func=flow_distance_other_flow_matrix_weighted_with_length)),
        ODAlgorithmInfo(name="FlowDBSCAN (FDS)", func=flow_cluster_DBSCAN,
                        other_func_kwargs=dict(eps=0.09, min_flows=5, n_jobs=-1, flow_distance_func=flow_distance_matrix_weighted_with_length)),
        ODAlgorithmInfo(name="ScaleFC", func=flow_cluster_ScaleFC, other_func_kwargs=dict(
            scale_factor=0.205, min_flows=9, n_jobs=-1)),
    ],

    "C": [
        ODAlgorithmInfo(name="FlowLF", func=flow_cluster_LF,
                        other_func_kwargs=dict(lambda_=1, radius=3, n_jobs=-1)),
        ODAlgorithmInfo(name="FlowDBSCAN", func=flow_cluster_DBSCAN,
                        other_func_kwargs=dict(eps=2.3, min_flows=5, n_jobs=-1)),
        ODAlgorithmInfo(name="FlowLF (SF)", func=flow_cluster_LF_with_scale_factor,
                        other_func_kwargs=dict(scale_factor=0.20, n_jobs=-1)),
        ODAlgorithmInfo(name="FlowDBSCAN (SF)", func=flow_cluster_DBSCAN_with_scale_factor,
                        other_func_kwargs=dict(scale_factor=0.19, min_flows=6, n_jobs=-1)),
        ODAlgorithmInfo(name="FlowLF (FDS)", func=flow_cluster_LF,
                        other_func_kwargs=dict(lambda_=1, radius=0.14, n_jobs=-1, flow_distance_func=flow_distance_matrix_weighted_with_length, flow_distance_other_flow_func=flow_distance_other_flow_matrix_weighted_with_length)),
        ODAlgorithmInfo(name="FlowDBSCAN (FDS)", func=flow_cluster_DBSCAN,
                        other_func_kwargs=dict(eps=0.1, min_flows=6, n_jobs=-1, flow_distance_func=flow_distance_matrix_weighted_with_length)),
        ODAlgorithmInfo(name="ScaleFC", func=flow_cluster_ScaleFC, other_func_kwargs=dict(
            scale_factor=0.19, min_flows=5, n_jobs=-1)),
    ],

    "D": [
        ODAlgorithmInfo(name="FlowLF", func=flow_cluster_LF,
                        other_func_kwargs=dict(lambda_=1, radius=6, n_jobs=-1)),
        ODAlgorithmInfo(name="FlowDBSCAN", func=flow_cluster_DBSCAN,
                        other_func_kwargs=dict(eps=4.5, min_flows=5, n_jobs=-1)),
        ODAlgorithmInfo(name="FlowLF (SF)", func=flow_cluster_LF_with_scale_factor,
                        other_func_kwargs=dict(scale_factor=0.24, n_jobs=-1)),
        ODAlgorithmInfo(name="FlowDBSCAN (SF)", func=flow_cluster_DBSCAN_with_scale_factor,
                        other_func_kwargs=dict(scale_factor=0.24, min_flows=5, n_jobs=-1)),
        ODAlgorithmInfo(name="FlowLF (FDS)", func=flow_cluster_LF,
                        other_func_kwargs=dict(lambda_=1, radius=0.14, n_jobs=-1, flow_distance_func=flow_distance_matrix_weighted_with_length, flow_distance_other_flow_func=flow_distance_other_flow_matrix_weighted_with_length)),
        ODAlgorithmInfo(name="FlowDBSCAN (FDS)", func=flow_cluster_DBSCAN,
                        other_func_kwargs=dict(eps=0.11, min_flows=5, n_jobs=-1, flow_distance_func=flow_distance_matrix_weighted_with_length)),
        ODAlgorithmInfo(name="ScaleFC", func=flow_cluster_ScaleFC, other_func_kwargs=dict(
            scale_factor=0.24, min_flows=5, n_jobs=-1))
    ],

    "E": [
        ODAlgorithmInfo(name="FlowLF", func=flow_cluster_LF,
                        other_func_kwargs=dict(lambda_=1, radius=6.5, n_jobs=-1)),
        ODAlgorithmInfo(name="FlowDBSCAN", func=flow_cluster_DBSCAN,
                        other_func_kwargs=dict(eps=2.8, min_flows=5, n_jobs=-1)),
        ODAlgorithmInfo(name="FlowLF (SF)", func=flow_cluster_LF_with_scale_factor,
                        other_func_kwargs=dict(scale_factor=0.16, n_jobs=-1)),
        ODAlgorithmInfo(name="FlowDBSCAN (SF)", func=flow_cluster_DBSCAN_with_scale_factor,
                        other_func_kwargs=dict(scale_factor=0.16, min_flows=5, n_jobs=-1)),
        ODAlgorithmInfo(name="FlowLF (FDS)", func=flow_cluster_LF,
                        other_func_kwargs=dict(lambda_=1, radius=0.12, n_jobs=-1, flow_distance_func=flow_distance_matrix_weighted_with_length, flow_distance_other_flow_func=flow_distance_other_flow_matrix_weighted_with_length)),
        ODAlgorithmInfo(name="FlowDBSCAN (FDS)", func=flow_cluster_DBSCAN,
                        other_func_kwargs=dict(eps=0.1, min_flows=5, n_jobs=-1, flow_distance_func=flow_distance_matrix_weighted_with_length)),
        ODAlgorithmInfo(name="ScaleFC", func=flow_cluster_ScaleFC, other_func_kwargs=dict(
            scale_factor=0.23, min_flows=5, n_jobs=-1, can_discard_flow_group_func=lambda x, y: len(x) < 8)),
    ],

    "F": [
        ODAlgorithmInfo(name="FlowLF", func=flow_cluster_LF,
                        other_func_kwargs=dict(lambda_=1, radius=6, n_jobs=-1)),
        ODAlgorithmInfo(name="FlowDBSCAN", func=flow_cluster_DBSCAN,
                        other_func_kwargs=dict(eps=3.8, min_flows=5, n_jobs=-1)),
        ODAlgorithmInfo(name="FlowLF (SF)", func=flow_cluster_LF_with_scale_factor,
                        other_func_kwargs=dict(scale_factor=0.17, n_jobs=-1)),
        ODAlgorithmInfo(name="FlowDBSCAN (SF)", func=flow_cluster_DBSCAN_with_scale_factor,
                        other_func_kwargs=dict(scale_factor=0.21, min_flows=5, n_jobs=-1)),
        ODAlgorithmInfo(name="FlowLF (FDS)", func=flow_cluster_LF,
                        other_func_kwargs=dict(lambda_=1, radius=0.1, n_jobs=-1, flow_distance_func=flow_distance_matrix_weighted_with_length, flow_distance_other_flow_func=flow_distance_other_flow_matrix_weighted_with_length)),
        ODAlgorithmInfo(name="FlowDBSCAN (FDS)", func=flow_cluster_DBSCAN,
                        other_func_kwargs=dict(eps=0.11, min_flows=5, n_jobs=-1, flow_distance_func=flow_distance_matrix_weighted_with_length)),
        ODAlgorithmInfo(name="ScaleFC", func=flow_cluster_ScaleFC, other_func_kwargs=dict(
            scale_factor=0.205, min_flows=5, n_jobs=-1, can_discard_flow_group_func=lambda x, y: len(x) < 8))
    ],
}


reordered_all_data_algo_infos = {
    "A": [
        ODAlgorithmInfo(name="FlowLF", func=flow_cluster_LF,
                        other_func_kwargs=dict(lambda_=1, radius=3, n_jobs=-1)),
        ODAlgorithmInfo(name="FlowLF (SF)", func=flow_cluster_LF_with_scale_factor,
                        other_func_kwargs=dict(scale_factor=0.26, n_jobs=-1)),
        ODAlgorithmInfo(name="FlowLF (FDS)", func=flow_cluster_LF,
                        other_func_kwargs=dict(lambda_=1, radius=0.15, n_jobs=-1, flow_distance_func=flow_distance_matrix_weighted_with_length, flow_distance_other_flow_func=flow_distance_other_flow_matrix_weighted_with_length)),
        ODAlgorithmInfo(name="FlowDBSCAN", func=flow_cluster_DBSCAN,
                        other_func_kwargs=dict(eps=2.5, min_flows=5, n_jobs=-1)),
        ODAlgorithmInfo(name="FlowDBSCAN (SF)", func=flow_cluster_DBSCAN_with_scale_factor,
                        other_func_kwargs=dict(scale_factor=0.2, min_flows=5, n_jobs=-1)),
        ODAlgorithmInfo(name="FlowDBSCAN (FDS)", func=flow_cluster_DBSCAN,
                        other_func_kwargs=dict(eps=0.1, min_flows=5, n_jobs=-1, flow_distance_func=flow_distance_matrix_weighted_with_length)),
        ODAlgorithmInfo(name="ScaleFC", func=flow_cluster_ScaleFC, other_func_kwargs=dict(
            scale_factor=0.2, min_flows=4, n_jobs=-1, can_discard_flow_group_func=lambda x, y: len(x) < 8))
    ],

    "B": [
        ODAlgorithmInfo(name="FlowLF", func=flow_cluster_LF,
                        other_func_kwargs=dict(lambda_=1, radius=2, n_jobs=-1)),
        ODAlgorithmInfo(name="FlowLF (SF)", func=flow_cluster_LF_with_scale_factor,
                        other_func_kwargs=dict(scale_factor=0.23, n_jobs=-1)),
        ODAlgorithmInfo(name="FlowLF (FDS)", func=flow_cluster_LF,
                        other_func_kwargs=dict(lambda_=1, radius=0.13, n_jobs=-1, flow_distance_func=flow_distance_matrix_weighted_with_length, flow_distance_other_flow_func=flow_distance_other_flow_matrix_weighted_with_length)),
        ODAlgorithmInfo(name="FlowDBSCAN", func=flow_cluster_DBSCAN,
                        other_func_kwargs=dict(eps=1.6, min_flows=5, n_jobs=-1)),
        ODAlgorithmInfo(name="FlowDBSCAN (SF)", func=flow_cluster_DBSCAN_with_scale_factor,
                        other_func_kwargs=dict(scale_factor=0.18, min_flows=5, n_jobs=-1)),
        ODAlgorithmInfo(name="FlowDBSCAN (FDS)", func=flow_cluster_DBSCAN,
                        other_func_kwargs=dict(eps=0.09, min_flows=5, n_jobs=-1, flow_distance_func=flow_distance_matrix_weighted_with_length)),
        ODAlgorithmInfo(name="ScaleFC", func=flow_cluster_ScaleFC, other_func_kwargs=dict(
            scale_factor=0.205, min_flows=9, n_jobs=-1)),
    ],

    "C": [
        ODAlgorithmInfo(name="FlowLF", func=flow_cluster_LF,
                        other_func_kwargs=dict(lambda_=1, radius=3, n_jobs=-1)),
        ODAlgorithmInfo(name="FlowLF (SF)", func=flow_cluster_LF_with_scale_factor,
                        other_func_kwargs=dict(scale_factor=0.20, n_jobs=-1)),
        ODAlgorithmInfo(name="FlowLF (FDS)", func=flow_cluster_LF,
                        other_func_kwargs=dict(lambda_=1, radius=0.14, n_jobs=-1, flow_distance_func=flow_distance_matrix_weighted_with_length, flow_distance_other_flow_func=flow_distance_other_flow_matrix_weighted_with_length)),
        ODAlgorithmInfo(name="FlowDBSCAN", func=flow_cluster_DBSCAN,
                        other_func_kwargs=dict(eps=2.3, min_flows=5, n_jobs=-1)),
        ODAlgorithmInfo(name="FlowDBSCAN (SF)", func=flow_cluster_DBSCAN_with_scale_factor,
                        other_func_kwargs=dict(scale_factor=0.19, min_flows=6, n_jobs=-1)),
        ODAlgorithmInfo(name="FlowDBSCAN (FDS)", func=flow_cluster_DBSCAN,
                        other_func_kwargs=dict(eps=0.1, min_flows=6, n_jobs=-1, flow_distance_func=flow_distance_matrix_weighted_with_length)),
        ODAlgorithmInfo(name="ScaleFC", func=flow_cluster_ScaleFC, other_func_kwargs=dict(
            scale_factor=0.19, min_flows=5, n_jobs=-1)),
    ],

    "D": [
        ODAlgorithmInfo(name="FlowLF", func=flow_cluster_LF,
                        other_func_kwargs=dict(lambda_=1, radius=6, n_jobs=-1)),
        ODAlgorithmInfo(name="FlowLF (SF)", func=flow_cluster_LF_with_scale_factor,
                        other_func_kwargs=dict(scale_factor=0.24, n_jobs=-1)),
        ODAlgorithmInfo(name="FlowLF (FDS)", func=flow_cluster_LF,
                        other_func_kwargs=dict(lambda_=1, radius=0.14, n_jobs=-1, flow_distance_func=flow_distance_matrix_weighted_with_length, flow_distance_other_flow_func=flow_distance_other_flow_matrix_weighted_with_length)),
        ODAlgorithmInfo(name="FlowDBSCAN", func=flow_cluster_DBSCAN,
                        other_func_kwargs=dict(eps=4.5, min_flows=5, n_jobs=-1)),
        ODAlgorithmInfo(name="FlowDBSCAN (SF)", func=flow_cluster_DBSCAN_with_scale_factor,
                        other_func_kwargs=dict(scale_factor=0.24, min_flows=5, n_jobs=-1)),
        ODAlgorithmInfo(name="FlowDBSCAN (FDS)", func=flow_cluster_DBSCAN,
                        other_func_kwargs=dict(eps=0.11, min_flows=5, n_jobs=-1, flow_distance_func=flow_distance_matrix_weighted_with_length)),
        ODAlgorithmInfo(name="ScaleFC", func=flow_cluster_ScaleFC, other_func_kwargs=dict(
            scale_factor=0.24, min_flows=5, n_jobs=-1))
    ],

    "E": [
        ODAlgorithmInfo(name="FlowLF", func=flow_cluster_LF,
                        other_func_kwargs=dict(lambda_=1, radius=6.5, n_jobs=-1)),
        ODAlgorithmInfo(name="FlowLF (SF)", func=flow_cluster_LF_with_scale_factor,
                        other_func_kwargs=dict(scale_factor=0.16, n_jobs=-1)),
        ODAlgorithmInfo(name="FlowLF (FDS)", func=flow_cluster_LF,
                        other_func_kwargs=dict(lambda_=1, radius=0.12, n_jobs=-1, flow_distance_func=flow_distance_matrix_weighted_with_length, flow_distance_other_flow_func=flow_distance_other_flow_matrix_weighted_with_length)),
        ODAlgorithmInfo(name="FlowDBSCAN", func=flow_cluster_DBSCAN,
                        other_func_kwargs=dict(eps=2.8, min_flows=5, n_jobs=-1)),
        ODAlgorithmInfo(name="FlowDBSCAN (SF)", func=flow_cluster_DBSCAN_with_scale_factor,
                        other_func_kwargs=dict(scale_factor=0.16, min_flows=5, n_jobs=-1)),
        ODAlgorithmInfo(name="FlowDBSCAN (FDS)", func=flow_cluster_DBSCAN,
                        other_func_kwargs=dict(eps=0.1, min_flows=5, n_jobs=-1, flow_distance_func=flow_distance_matrix_weighted_with_length)),
        ODAlgorithmInfo(name="ScaleFC", func=flow_cluster_ScaleFC, other_func_kwargs=dict(
            scale_factor=0.23, min_flows=5, n_jobs=-1, can_discard_flow_group_func=lambda x, y: len(x) < 8)),
    ],

    "F": [
        ODAlgorithmInfo(name="FlowLF", func=flow_cluster_LF,
                        other_func_kwargs=dict(lambda_=1, radius=6, n_jobs=-1)),
        ODAlgorithmInfo(name="FlowLF (SF)", func=flow_cluster_LF_with_scale_factor,
                        other_func_kwargs=dict(scale_factor=0.17, n_jobs=-1)),
        ODAlgorithmInfo(name="FlowLF (FDS)", func=flow_cluster_LF,
                        other_func_kwargs=dict(lambda_=1, radius=0.1, n_jobs=-1, flow_distance_func=flow_distance_matrix_weighted_with_length, flow_distance_other_flow_func=flow_distance_other_flow_matrix_weighted_with_length)),
        ODAlgorithmInfo(name="FlowDBSCAN", func=flow_cluster_DBSCAN,
                        other_func_kwargs=dict(eps=3.8, min_flows=5, n_jobs=-1)),
        ODAlgorithmInfo(name="FlowDBSCAN (SF)", func=flow_cluster_DBSCAN_with_scale_factor,
                        other_func_kwargs=dict(scale_factor=0.21, min_flows=5, n_jobs=-1)),
        ODAlgorithmInfo(name="FlowDBSCAN (FDS)", func=flow_cluster_DBSCAN,
                        other_func_kwargs=dict(eps=0.12, min_flows=5, n_jobs=-1, flow_distance_func=flow_distance_matrix_weighted_with_length)),
        ODAlgorithmInfo(name="ScaleFC", func=flow_cluster_ScaleFC, other_func_kwargs=dict(
            scale_factor=0.205, min_flows=5, n_jobs=-1, can_discard_flow_group_func=lambda x, y: len(x) < 8))
    ],
}


def run_four_algorithms(save=True):
    # 执行所有的算法，并输出df
    df = pd.DataFrame(columns=["DataName", "AlgoName",
                      "RealLabel", "PredLabel", "ARI"])

    for name, algo_info in reordered_all_data_algo_infos.items():
        data_name = f"Dataset {name}"
        filepath = f'./data/Data{name}.txt'
        labelpath = f'./data/Data{name}-label.txt'
        OD = np.loadtxt(filepath, delimiter=',')
        real_label = np.loadtxt(labelpath, delimiter=',', dtype=int)
        for algo in algo_info:
            algo_name = algo.name
            func_kwargs = algo.other_func_kwargs
            func = algo.func
            func_kwargs["OD"] = OD
            pred_label = func(**func_kwargs)
            ARI = ClusterExternalIndexAny(real_label, pred_label).ARI

            df.loc[len(df)] = [data_name, algo_name,
                               real_label, pred_label, ARI]
    if save:
        df.to_csv("./result/synthetic_clustering_resuls_with_sf_and_FDS.csv", index=False)


if __name__ == "__main__":
    run_four_algorithms(save=True)

    