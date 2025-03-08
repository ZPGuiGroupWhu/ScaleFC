
import sys
sys.path.append(".")

from algorithm.FlowLF import flow_cluster_LF
from algorithm.ScaleFC import flow_cluster_ScaleFC
from algorithm.FlowDBSCAN import flow_cluster_DBSCAN
from algorithm.AFC import flow_cluster_AFC
from algorithm.util_tools import *
from dataclasses import dataclass

from typing import Callable
import pandas as pd
from sklearn import metrics



class ClusterExternalIndexAny:
    def __init__(self, real_lable, pred_label) -> None:
        self._real_label = np.asarray(real_lable, dtype=int)
        self._pred_label = np.asarray(pred_label, dtype=int)
        assert len(real_lable) == len(pred_label)
        assert len(real_lable) > 0

    @property
    def ARI(self):
        if np.all(self._pred_label == -1):
            return 0
        return metrics.adjusted_rand_score(self._real_label, self._pred_label)


@dataclass
class ODAlgorithmInfo:
    name: str  # name of the algorithm
    func: Callable
    other_func_kwargs: dict  # func(OD=OD, **func_kwargs)


all_data_algo_infos = {
    "A": [
        ODAlgorithmInfo(name="AFC", func=flow_cluster_AFC, other_func_kwargs=dict(
            k=44, min_num_of_subcluster=20, n_jobs=-1)),
        ODAlgorithmInfo(name="FlowLF", func=flow_cluster_LF,
                        other_func_kwargs=dict(lambda_=1, radius=3, n_jobs=-1)),
        ODAlgorithmInfo(name="FlowDBSCAN", func=flow_cluster_DBSCAN,
                        other_func_kwargs=dict(eps=2.5, min_flows=5, n_jobs=-1)),
        ODAlgorithmInfo(name="ScaleFC", func=flow_cluster_ScaleFC, other_func_kwargs=dict(
            scale_factor=0.2, min_flows=4, n_jobs=-1, can_discard_flow_group_func=lambda x, y: len(x) < 8)),
        ODAlgorithmInfo(name="ScaleFC (Adapted)", func=flow_cluster_ScaleFC, other_func_kwargs=dict(
            scale_factor=0.2, min_flows=7, n_jobs=-1, can_discard_flow_group_func=lambda x, y: len(x) < 8)),
    ],

    "B": [
        ODAlgorithmInfo(name="AFC", func=flow_cluster_AFC, other_func_kwargs=dict(
            k=48, min_num_of_subcluster=20, n_jobs=-1)),
        ODAlgorithmInfo(name="FlowLF", func=flow_cluster_LF,
                        other_func_kwargs=dict(lambda_=1, radius=2, n_jobs=-1)),
        ODAlgorithmInfo(name="FlowDBSCAN", func=flow_cluster_DBSCAN,
                        other_func_kwargs=dict(eps=1.6, min_flows=5, n_jobs=-1)),
        ODAlgorithmInfo(name="ScaleFC", func=flow_cluster_ScaleFC, other_func_kwargs=dict(
            scale_factor=0.2, min_flows=9, n_jobs=-1)),
        ODAlgorithmInfo(name="ScaleFC (Adapted)", func=flow_cluster_ScaleFC, other_func_kwargs=dict(
            scale_factor=0.2, min_flows=7, n_jobs=-1)),
    ],

    "C": [
        ODAlgorithmInfo(name="AFC", func=flow_cluster_AFC, other_func_kwargs=dict(
            k=52, min_num_of_subcluster=12, n_jobs=-1)),
        ODAlgorithmInfo(name="FlowLF", func=flow_cluster_LF,
                        other_func_kwargs=dict(lambda_=1, radius=3, n_jobs=-1)),
        ODAlgorithmInfo(name="FlowDBSCAN", func=flow_cluster_DBSCAN,
                        other_func_kwargs=dict(eps=2.3, min_flows=5, n_jobs=-1)),
        ODAlgorithmInfo(name="ScaleFC", func=flow_cluster_ScaleFC, other_func_kwargs=dict(
            scale_factor=0.19, min_flows=5, n_jobs=-1)),
        ODAlgorithmInfo(name="ScaleFC (Adapted)", func=flow_cluster_ScaleFC, other_func_kwargs=dict(
            scale_factor=0.23, min_flows=5, n_jobs=-1)),
    ],

    "D": [
        ODAlgorithmInfo(name="AFC", func=flow_cluster_AFC, other_func_kwargs=dict(
            k=29, min_num_of_subcluster=20, n_jobs=-1)),
        ODAlgorithmInfo(name="FlowLF", func=flow_cluster_LF,
                        other_func_kwargs=dict(lambda_=1, radius=6, n_jobs=-1)),
        ODAlgorithmInfo(name="FlowDBSCAN", func=flow_cluster_DBSCAN,
                        other_func_kwargs=dict(eps=4.5, min_flows=5, n_jobs=-1)),
        ODAlgorithmInfo(name="ScaleFC", func=flow_cluster_ScaleFC, other_func_kwargs=dict(
            scale_factor=0.24, min_flows=5, n_jobs=-1)),
        ODAlgorithmInfo(name="ScaleFC (Adapted)", func=flow_cluster_ScaleFC, other_func_kwargs=dict(
            scale_factor=0.24, min_flows=8, n_jobs=-1)),
    ],

    "E": [
        ODAlgorithmInfo(name="AFC", func=flow_cluster_AFC, other_func_kwargs=dict(
            k=34, min_num_of_subcluster=12, n_jobs=-1)),
        ODAlgorithmInfo(name="FlowLF", func=flow_cluster_LF,
                        other_func_kwargs=dict(lambda_=1, radius=6.5, n_jobs=-1)),
        ODAlgorithmInfo(name="FlowDBSCAN", func=flow_cluster_DBSCAN,
                        other_func_kwargs=dict(eps=2.8, min_flows=5, n_jobs=-1)),
        ODAlgorithmInfo(name="ScaleFC", func=flow_cluster_ScaleFC, other_func_kwargs=dict(
            scale_factor=0.23, min_flows=5, n_jobs=-1, can_discard_flow_group_func=lambda x, y: len(x) < 8)),
        ODAlgorithmInfo(name="ScaleFC (Adapted)", func=flow_cluster_ScaleFC, other_func_kwargs=dict(
            scale_factor=0.24, min_flows=8, n_jobs=-1, can_discard_flow_group_func=lambda x, y: len(x) < 8)),
    ],

    "F": [
        ODAlgorithmInfo(name="AFC", func=flow_cluster_AFC, other_func_kwargs=dict(
            k=38, min_num_of_subcluster=11, n_jobs=-1)),
        ODAlgorithmInfo(name="FlowLF", func=flow_cluster_LF,
                        other_func_kwargs=dict(lambda_=1, radius=6, n_jobs=-1)),
        ODAlgorithmInfo(name="FlowDBSCAN", func=flow_cluster_DBSCAN,
                        other_func_kwargs=dict(eps=3.8, min_flows=5, n_jobs=-1)),
        ODAlgorithmInfo(name="ScaleFC", func=flow_cluster_ScaleFC, other_func_kwargs=dict(
            scale_factor=0.205, min_flows=5, n_jobs=-1, can_discard_flow_group_func=lambda x, y: len(x) < 8)),
        ODAlgorithmInfo(name="ScaleFC (Adapted)", func=flow_cluster_ScaleFC, other_func_kwargs=dict(
            scale_factor=0.25, min_flows=6, n_jobs=-1, can_discard_flow_group_func=lambda x, y: len(x) < 8)),
    ],
}


def run_four_algorithms(save=True):
    # 执行所有的算法，并输出df
    df = pd.DataFrame(columns=["DataName", "AlgoName",
                      "RealLabel", "PredLabel", "ARI"])

    for name, algo_info in all_data_algo_infos.items():
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
        df.to_csv("./result/synthetic_clustering_resuls.csv", index=False)


if __name__ == "__main__":
    run_four_algorithms(save=True)
