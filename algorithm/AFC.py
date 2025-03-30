"""
Agglomerative hierarchical flow clustering algorithm.

Input: 
   T = {Ti | 1 ≤ i ≤ n} – a set of origin-destination flows 
   k – the number of nearest neighbors used in calculating distance.
Output: 
   A set of flow clusters C = {Cl | 1 < l << n}
Steps:
    1. Identify neighboring flows for each flow with a search radius k and create contiguous flow pairs, as explained in Section 3.2.
    2. Calculate the distance for each contiguity flow pair according to Equation 1, as explained in Section 3.3;
    3. Sort all contiguous flow pairs to an ascending order based on their distances;
    4. Initialize a set of flow clusters by making each flow a unique cluster, i.e. C = {Cl} and Cl = {Ti}, 1 ≤ l ≤ n; and
    5. For each contiguity flow pair (p, q), following the above ascending order:
        a. Find the two clusters Cx and Cy that p and q belong to: p ∈ Cx and q ∈ Cy;
        b. Calculate the distance dist(Cx, Cy) between Cx and Cy (see text below for detail); and
        c. If x ≠ y and dist(Cx, Cy) < 1, merge them: Cx = Cx∪Cy and C = C\Cy

Paper Reference: Zhu, Xi, and Diansheng Guo. “Mapping Large Spatial Flow Data with Hierarchical Clustering.” Transactions in GIS 18, no. 3 (June 2014): 421–35. https://doi.org/10.1111/tgis.12100.

How to use:
    from algorithm.AFC import flow_cluster_AFC
    OD = ...
    # to specify k manually
    label = flow_cluster_AFC(OD, k=5) 
    # to determin k by the condition that at least 95% flows have 1 neighbor and at least 70% of flows have 7 neighbors.
    label = flow_cluster_AFC(OD, k=None, determin_k_by_m=True, at_least_m=(1, 7), at_least_ratio=(0.95, 0.7)) 
"""

from typing import Iterable, Optional, Union
import numpy as np
from joblib import Parallel, delayed
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from algorithm.util_tools import *


def flow_cluster_AFC(
    OD: np.ndarray,
    k: Optional[int] = None,
    min_num_of_subcluster: Optional[int] = None,
    *,
    n_jobs: Optional[int] = None,
    batch_size: int = 100,
    return_origin_label: bool = False,
    determin_k_by_m: bool = False,
    at_least_m: Optional[Iterable] = None,
    at_least_ratio: Optional[Iterable] = None,
) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """AFC algorithm

    Args:
        OD (np.ndarray): OD matrix, shape must be N x 4, the four columns are ox, oy, dx, dy.
        k (int, optional): Number of neighbor flows. Defaults to None.
        min_num_of_subcluster (Optional[int], optional): if None, do not filter subclusters; if int, must be greater than 0. Defaults to None.
        n_jobs (Optional[int], optional): number of jobs to run in parallel. Defaults to None.
        batch_size (int, optional): batch size for knn search. Defaults to 100.
        return_origin_label (bool, optional): if True and min_num_of_subcluster is not None, return the origin label, otherwise return the filtered label only. Defaults to False.
        determin_k_by_m (bool, optional): if True, determine k by m. Defaults to False.
        at_least_m (Optional[Iterable], optional): If determin_k_by_m is True, it must be Iterable and every item's type is Int. Defaults to None.
        at_least_ratio (Optional[Iterable], optional): If determin_k_by_m is True, the length must be equal to list(at_least_m), and every item is less that 1 and larger than 0. Defaults to None.
    Returns:
        np.ndarray: label or (label, origin_label)
    """
    flow_check_OD(OD)
    assert min_num_of_subcluster is None or (isinstance(
        min_num_of_subcluster, int) and min_num_of_subcluster > 0), f"min_num_of_subcluster must be greater than 0"
    if min_num_of_subcluster is None:
        print(f"min_num_of_subcluster is None, will generate subclusters with few flows.")
    if not determin_k_by_m:
        assert isinstance(k, int) and k > 0, f"k must be greater than 0"
    else:
        assert isinstance(at_least_m, Iterable) and len(
            at_least_m) > 0, f"at_least_m must be a non-empty iterable"
        assert isinstance(at_least_ratio, Iterable) and len(at_least_ratio) == len(
            at_least_m), f"at_least_ratio must be a iterable with the same length as at_least_m"
        at_least_m = tuple(at_least_m)
        at_least_ratio = tuple(at_least_ratio)
        assert all(isinstance(i, int) and i >
                   0 for i in at_least_m), f"at_least_m must be a iterable with all items greater than 0"
        assert all(isinstance(i, float) and 0 < i <
                   1 for i in at_least_ratio), f"at_least_ratio must be a iterable with all items greater than 0 and less than 1"

        if k:
            print(
                f"determin_k_by_m is True, k={k} will not take effect and it will be determined by at_least_m and at_least_ratio setting.")
        k = _afc_determine_k_from_m_dist(
            OD, at_least_m, at_least_ratio, n_jobs, batch_size)

    dm = _snn(OD, k)

    label = AgglomerativeClustering(
        n_clusters=None, linkage='complete', metric="precomputed", distance_threshold=1).fit_predict(dm)
    del dm
    if min_num_of_subcluster:
        origin_label = np.copy(label)
        label = clusters_relabel_by_number_of_each_subcluster(
            label, min_num_of_subcluster)
        if return_origin_label:
            return label, origin_label
    return label

def _knn_intersection_matrix(points, k):

    # 计算 KNN（排除自身）
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto', n_jobs=-1).fit(points)
    _, indices = nbrs.kneighbors(points)
    knn_indices = indices[:, 1:]  # 移除第一个元素（自身）

    # 构建稀疏邻接矩阵
    N = points.shape[0]
    rows = np.repeat(np.arange(N), k)
    cols = knn_indices.ravel()
    adj = csr_matrix((np.ones_like(rows), (rows, cols)), shape=(N, N), dtype=np.int32)

    # 矩阵乘法计算交集数量 (利用稀疏矩阵优化)
    intersection = adj.dot(adj.T)

    # 转换为密集数组并对称化
    return (intersection + intersection.T).toarray() // 2  # 消除重复计数


def _snn(OD, k):
    op,dp = flow_OD_points(OD)
    return 1 - _knn_intersection_matrix(op, k) * _knn_intersection_matrix(dp, k) / k / k

def _afc_flow_knn_length(OD, k, n_jobs=None, batch_size=100) -> list:
    op, dp = flow_OD_points(OD)
    op_knn_indices = point_knn(op, k)
    dp_knn_indices = point_knn(dp, k)

    if n_jobs:
        assert isinstance(n_jobs, int), f"n_jobs must be int, but got {n_jobs}"

        def compute_batch(batch_indices):
            batch_results = []
            for i in batch_indices:
                intersec = np.intersect1d(
                    op_knn_indices[i], dp_knn_indices[i], assume_unique=True)
                batch_results.append(intersec.size)
            return batch_results

        # 生成批次索引
        indices = list(range(len(op_knn_indices)))
        batches = [indices[i: i + batch_size]
                   for i in range(0, len(indices), batch_size)]

        # 使用 joblib 进行并行计算
        flow_knn_indices_length = Parallel(n_jobs=n_jobs)(
            delayed(compute_batch)(batch) for batch in batches)

        # 展开结果
        flow_knn_indices_length = [
            item for sublist in flow_knn_indices_length for item in sublist]

        return flow_knn_indices_length
    else:
        flow_knn_indices_length = []
        for i in range(len(op_knn_indices)):
            intersec = np.intersect1d(
                op_knn_indices[i], dp_knn_indices[i], assume_unique=True)
            flow_knn_indices_length.append(intersec.size)

        return flow_knn_indices_length


# 根据 m 来推定 k
# 比如：at_least_m = 5, at_least_ratio=0.95 表示至少有 95% 的流有 5 个邻居，输出此时最小的 k
# at_least_m = [5, 1], at_least_ratio=[0.7, 0.95]， 表示至少有 95% 的流有 1 个邻居；并且至少有 70% 的流有 1 个邻居，输出此时最小的 k
def _afc_determine_k_from_m_dist(OD: np.ndarray, at_least_m, at_least_ratio, n_jobs, batch_size) -> int:
    N = flow_number(OD)
    at_least_neighbors = tuple(int(x * N) for x in at_least_ratio)

    low = np.max(at_least_m)
    high = 2 * low
    k = high

    def _helper_this_k_is_ok(_k):
        _indices_len = np.asarray(_afc_flow_knn_length(
            OD, _k, n_jobs, batch_size), dtype=int)
        for x, y in zip(at_least_m, at_least_neighbors):
            if np.sum(_indices_len >= x) < y:
                return False
        return True

    while True:
        if _helper_this_k_is_ok(k):
            break
        else:
            low = high
            high = 2 * high
            k = high

    # 现在来寻找最小的k
    while low < high:
        mid = (low + high) // 2
        if _helper_this_k_is_ok(mid):
            high = mid
            k = mid
        else:
            low = mid + 1
    if k > N:
        print("The determined k is too large and it's lager than the number of flow, k is: {}".format(k))

    msg = f"The best k for conditions below is {k}, the conditions are: "
    ccount = 1
    for x, y in zip(at_least_m, at_least_ratio):
        msg += f"{ccount}. At least {100 * y:.2f}% flow(s) have {x} neighbor(s). "
    # _log.debug(msg)
    return k
