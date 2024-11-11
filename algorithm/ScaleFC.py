"""
ScaleFC: a scale-aware flow clustering algorithm

Steps:
    (1) Eliminate noise flows and identify flow groups via spatial connectivity measurement; 
    (2) Recognize the strongly-connected flow groups using the spatial compactness indicator. The strongly-connected flow groups are assigned as the final clusters, while the remain groups are treated as weakly-connected flow groups; 
    (3) Identify partitioning flows (PFs) within the generated weakly-connected flow groups to detect potential strongly-connected groups; 
    (4) Reallocate all partitioning flows to nearest flow clusters and output cluster results.

How to use:
    from algorithm.ScaleFC import flow_cluster_ScaleFC
    OD = ...
    # to specify k manually
    label = flow_cluster_ScaleFC(OD, scale_factor=0.1, min_flows=5)
"""

from collections import deque
from datetime import datetime
import time
from typing import Callable, Literal, Optional, Union

import numpy as np
from joblib import Parallel, delayed
from sklearn.neighbors import NearestNeighbors

from algorithm.util_tools import *

ASSERT_ENABLED = False
DEBUG_ENABLED = False


def flow_cluster_ScaleFC(
    OD: np.ndarray,
    scale_factor: Optional[float] = 0.1,
    min_flows: int = 5,
    eps: Optional[float] = None,
    n_jobs: Optional[int] = None,
    get_steps_time: bool = False, **kwargs
) -> np.ndarray[int]:
    """clustering by detecting prititioning flow for spatial-connected flow groups.

    Args:
        OD (np.ndarray): OD matrix has 4 columns:  ox, oy, dx, dy.
        scale_factor (float): scale factor, 0-1.
        min_flows (int, optional): minimum flows for a flow's neighbor. Defaults to 5.
        eps (float, optional): if eps is specified, it will be used to detect flow's neighbors and clusters, and scale factor will be disabled. Defaults to None.
        n_jobs (int, optional): number of jobs to run in parallel. Defaults to None. -1 means using all processors.
        kwargs:
            spatial_connected_groups_label (np.ndarray): label of spatial connected groups. If it is specified, the algorithm will use it to detect flow's neighbors and clusters. Defaults to None.

            flow_distance_func (Callable[[np.ndarray, str], np.ndarray]): function for calculating flow distance matrix.

            flow_distance_other_flow_func (Callable[[np.ndarray, np.ndarray, str], np.ndarray]): function for calculating flow distance matrix between two flow OD matrix.

            scale_factor_func (Union[Literal["linear", "square", "sqrt", "tanh"], Callable[[np.ndarray, float], np.ndarray]]): function for calcualting scale factor, return eps, the first param is flow OD matrix, the second param is scale factor.Defaults to _scale_factor_func_linear.

            is_strong_flow_group_func (Callable[[np.ndarray, *Any], bool]): function for determining to consider a flow group as strong or not, the first param is flow OD matrix, the other params is is_strong_flow_group_func_args. If strong, return True.
            is_strong_flow_group_func_args  (Union[tuple, list]): other args for is_strong_flow_group_func.

            can_discard_flow_group_func (Callable[[np.ndarray, *Any], bool]): function for determining to discard a flow group or not, the first param is flow OD matrix, the other params is can_discard_flow_group_func_args. If discard, return True.
            can_discard_flow_group_func_args (Union[tuple, list]): other args for can_discard_flow_group_func.

            calc_FKI_func (Callable[[np.ndarray], float]): function for calculating FKI, which is indicator of flow's k nearest neighboring subgroup, the only param is flow OD matrix.

            calc_FKID_func (Callable[[list], list]): function for calculating FKID, which is derivative of FKI, the only param is FKI.

            postprocess_pfs (bool, optional): whether to postprocess the partitioning flow. Defaults to True.

    """
    flow_check_OD(OD)
    assert (eps is None or eps >
            0) and min_flows > 0, f"Invalid params, eps: {eps}, scale_factor: {scale_factor}, min_flows: {min_flows}"

    kwargs_valid = [
        "spatial_connected_groups_label",
        "flow_distance_func",
        "flow_distance_other_flow_func",
        "scale_factor_func",
        "is_strong_flow_group_func",
        "is_strong_flow_group_func_args",
        "can_discard_flow_group_func",
        "can_discard_flow_group_func_args",
        "calc_FKI_func",
        "calc_FKID_func",
        "postprocess_pfs",
    ]
    for keys in kwargs.keys():
        if keys not in kwargs_valid:
            raise ValueError(f"Invalid kwargs: {keys}")

    spatial_connected_groups_label = kwargs.get(
        'spatial_connected_groups_label', None)

    flow_distance_func = kwargs.get(
        'flow_distance_func', flow_distance_matrix_max_euclidean)

    flow_distance_other_flow_func = kwargs.get(
        'flow_distance_other_flow_func', flow_distance_other_flow_matrix_max_euclidean)

    scale_factor_func = _check_scale_factor_func(
        kwargs.get('scale_factor_func', "linear"))

    if (is_strong_flow_group_func := kwargs.get('is_strong_flow_group_func', None)) is None:
        if eps is None:  # use scale_factor
            is_strong_flow_group_func = _is_strong_connected_cluster_with_scale_factor
        else:
            is_strong_flow_group_func = _is_strong_flow_group_with_fixed_eps

    if (is_strong_flow_group_func_args := kwargs.get('is_strong_flow_group_func_args', None)) is None:
        if eps is None:  # use scale_factor
            is_strong_flow_group_func_args = (scale_factor, scale_factor_func)
        else:
            is_strong_flow_group_func_args = (eps,)

    can_discard_flow_group_func_is_default = False
    if (can_discard_flow_group_func := kwargs.get('can_discard_flow_group_func', None)) is None:
        can_discard_flow_group_func = _spatial_connected_flow_groups_can_discard
        can_discard_flow_group_func_is_default = True

    can_discard_flow_group_func_args = kwargs.get(
        'can_discard_flow_group_func_args', (min_flows,))

    calc_FKI_func = kwargs.get(
        'calc_FKI_func', flow_indicator_rmse_OD_distance)

    calc_FKID_func = kwargs.get('calc_FKID_func', _calculate_FKID)

    postprocess_pfs = kwargs.get('postprocess_pfs', True)

    steps_time = [0, 0, 0, 0]

    OD_len = flow_number(OD)

    # process indices all the way
    # result, like: [[1, 3], [2, 5]], that means [1, 3] and [2, 5] are two clusetr respectively, [0, 4] are noises
    result_subclusters_indices = []
    all_pf_indices = []

    # step 1
    tt1 = time.time()

    # from dbscan or other methods
    if spatial_connected_groups_label is None:
        if eps is None:
            spatial_connected_groups_label = flow_label_spatial_connected_groups_with_scale_factor(
                OD, scale_factor, min_flows, flow_distance_func, scale_factor_func)
        else:
            spatial_connected_groups_label = flow_label_spatial_connected_groups_with_eps(
                flow_distance_func(OD), eps, min_flows)

    spatial_connected_groups_label = np.asarray(
        spatial_connected_groups_label, dtype=int)
    assert spatial_connected_groups_label.size == OD_len, "Invalid density_connected_clusters_label"

    # _labels, _counts = np.unique(spatial_connected_groups_label, return_counts=True)
    # _log.debug(f"spatial_connected_groups_label: \n{pd.DataFrame({'label': _labels, 'count': _counts}).to_string(index=False)}")

    subgroups_indices_queue = deque()
    for i in np.unique(spatial_connected_groups_label):
        if i < 0:
            continue
        idx = np.where(spatial_connected_groups_label == i)[0]
        if can_discard_flow_group_func_is_default and can_discard_flow_group_func(OD[idx, :], *can_discard_flow_group_func_args):
            continue
        subgroups_indices_queue.append(idx)
    # _log.debug(f"subgroups_indices_queue length: {len(subgroups_indices_queue)}")

    tt2 = time.time()
    steps_time[0] += tt2 - tt1  # 第一步的时间

    # use parallel or not, cannot use _log in parallel
    if not n_jobs:
        while subgroups_indices_queue:
            current_indices = subgroups_indices_queue.popleft()
            current_flow_group = OD[current_indices, :]

            tt1 = time.time()

            if can_discard_flow_group_func(current_flow_group, *can_discard_flow_group_func_args):
                # _log.debug(f"discard current flow group - current_indices: {current_indices}")
                tt2 = time.time()
                steps_time[1] += tt2 - tt1  # 第二步骤的时间
                continue

            if is_strong_flow_group_func(current_flow_group, *is_strong_flow_group_func_args):
                result_subclusters_indices.append(current_indices)
                # _log.debug(f"save current strong-connected flow group - current_indices: {current_indices}")
                tt2 = time.time()
                steps_time[1] += tt2 - tt1  # 第二步骤的时间
                continue

            tt2 = time.time()
            steps_time[1] += tt2 - tt1  # 第二步骤的时间

            # find pf and partition current flow group
            l, r, pf_idx = _flow_partition_arg(
                current_flow_group, min_flows, flow_distance_func, calc_FKI_func, calc_FKID_func, None)

            if ASSERT_ENABLED:
                curlist = [pf_idx] + l + r
                curlist.sort()
                assert np.array_equal(curlist, range(
                    len(current_indices))), "ops!"

            # _log.debug(f"find partitioning flow's index of input OD: {current_indices[pf_idx]}")

            if l:
                subgroups_indices_queue.append(
                    current_indices[np.asarray(l, dtype=int)])
                # print("left: ", l)
            if r:
                subgroups_indices_queue.append(
                    current_indices[np.asarray(r, dtype=int)])
                # print("right: ", r)
            if pf_idx:
                all_pf_indices.append(current_indices[pf_idx])

            tt3 = time.time()
            steps_time[2] += tt3 - tt2
    else:
        assert isinstance(n_jobs, int), f"n_jobs must be int, but got {n_jobs}"

        def process_subgroup(current_indices):
            current_flow_group = OD[current_indices, :]

            tt1 = time.time()

            if can_discard_flow_group_func(current_flow_group, *can_discard_flow_group_func_args):
                DEBUG_ENABLED and print(
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - DEBUG - [flow_cluster_dpfscf] - discard current flow group - current_indices: {current_indices}"
                )
                tt2 = time.time()
                # steps_time[1] += tt2 - tt1 # 第二步骤的时间
                return (None, None, tt2 - tt1)

            if is_strong_flow_group_func(current_flow_group, *is_strong_flow_group_func_args):
                DEBUG_ENABLED and print(
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - DEBUG - [flow_cluster_dpfscf] - save current strong-connected flow group - current_indices: {current_indices}"
                )
                tt2 = time.time()
                # steps_time[1] += tt2 - tt1 # 第二步骤的时间
                return (current_indices, None, tt2-tt1)

            tt2 = time.time()
            # steps_time[1] += tt2 - tt1 # 第二步骤的时间

            # find pf and partition current flow group
            l, r, pf_idx = _flow_partition_arg(
                current_flow_group, min_flows, flow_distance_func, calc_FKI_func, calc_FKID_func, n_jobs)

            if ASSERT_ENABLED:
                curlist = [pf_idx] + l + r
                curlist.sort()
                assert np.array_equal(curlist, range(
                    len(current_indices))), "ops!"

            DEBUG_ENABLED and print(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - DEBUG - [flow_cluster_dpfscf] - find partitioning flow's index of input OD: {current_indices[pf_idx]}"
            )

            left_indices = current_indices[np.asarray(
                l, dtype=int)] if l else None
            right_indices = current_indices[np.asarray(
                r, dtype=int)] if r else None
            pf_index = current_indices[pf_idx] if pf_idx else None

            tt3 = time.time()
            # steps_time[2] += tt3 - tt2 # 第三步骤的时间

            return (None, (left_indices, right_indices, pf_index), tt2-tt1, tt3-tt2)

        with Parallel(n_jobs=n_jobs) as parallel:
            while subgroups_indices_queue:
                # print(f"len(subgroups_indices_queue): {len(subgroups_indices_queue)}")
                results = parallel(delayed(process_subgroup)(
                    subgroups_indices_queue.popleft()) for _ in range(len(subgroups_indices_queue)))

                for result in results:
                    if result[0] is not None:
                        result_subclusters_indices.append(result[0])
                        # 只有第二步骤的时间
                        steps_time[1] += result[2]

                    if result[1] is not None:
                        left_indices, right_indices, pf_index = result[1]
                        if left_indices is not None:
                            subgroups_indices_queue.append(left_indices)
                        if right_indices is not None:
                            subgroups_indices_queue.append(right_indices)
                        if pf_index is not None:
                            all_pf_indices.append(pf_index)

                        steps_time[1] += result[2]  # 第二步都有
                        if len(result) == 4:
                            steps_time[2] += result[3]  # 第三步的加上都有

    tt3 = time.time()
    labels = np.full(OD_len, fill_value=-1, dtype=int)
    if not result_subclusters_indices:

        tt4 = time.time()
        steps_time[3] += tt4 - tt3

        if get_steps_time:
            return (labels, steps_time)
        return labels

    for i, x in enumerate(result_subclusters_indices):
        labels[x] = i

    if (not all_pf_indices) or (not postprocess_pfs):
        tt4 = time.time()
        steps_time[3] += tt4 - tt3
        if get_steps_time:
            return (labels, steps_time)

        return labels

    # process pd_index here
    all_pf_indices = np.asarray(all_pf_indices, dtype=int)
    # find cf for each group and calculate cf and pf's distance
    groups_centroid_flow = []
    for x in result_subclusters_indices:
        groups_centroid_flow.append(flow_centroid_OD(OD[x, :]))

    all_pf = OD[all_pf_indices, :]

    pf_cf_dis = flow_distance_other_flow_func(all_pf, groups_centroid_flow)

    pf_cf_dis_min_arg = np.argmin(pf_cf_dis, axis=1)
    assert len(all_pf_indices) == len(pf_cf_dis_min_arg), "ops!"

    for i, x in enumerate(pf_cf_dis_min_arg):
        cur_pf = all_pf[i, :]
        cur_group = OD[result_subclusters_indices[x], :]

        temp_flow = np.vstack((cur_group, cur_pf))
        if is_strong_flow_group_func(temp_flow, *is_strong_flow_group_func_args):
            labels[all_pf_indices[i]] = x

    tt4 = time.time()
    steps_time[3] += tt4 - tt3
    if get_steps_time:
        return (labels, steps_time)
    return labels


def flow_indicator_rmse_OD_distance(OD: np.ndarray) -> float:
    # 1. 计算中心流
    centroid_flow = flow_centroid_OD(OD)

    number = flow_number(OD)
    # 2. 计算每条流到中心流之间的距离
    max_dist = flow_distance_other_flow_matrix_max_euclidean(centroid_flow, OD)
    # print(max_dist.shape)
    # 3. 计算出最终的值
    res = np.sqrt(np.sum(np.square(max_dist)) / number)
    return res


def flow_indicator_rmse_length(OD):
    # 1. 计算中心流
    centroid_flow = flow_centroid_OD(OD)
    number = flow_number(OD)
    # 2. 计算每条流和中心流的距离
    cl = flow_length(centroid_flow)
    od_len = flow_length(OD)
    # 3. 计算出最终的值
    res = np.sqrt(np.sum(np.square(od_len - cl)) / number)
    return res


def flow_indicator_rmse_angle(OD) -> float:
    # 1. 计算中心流
    centroid_flow = flow_centroid_OD(OD)
    number = flow_number(OD)
    # 2. 计算每条流到中心流之间的夹角
    c_v = flow2vector(centroid_flow)
    # print(c_v)
    od_v = flow2vector(OD)
    # print(od_v)
    pro = np.dot(od_v, c_v)
    cos_s = pro / flow_length(centroid_flow) / flow_length(OD)
    cos_s = np.where(cos_s > 1, 1, np.where(cos_s < -1, -1, cos_s))
    res = np.arccos(cos_s)
    # 3. 计算出最终的值
    return np.sqrt(np.sum(np.square(res)) / number)

# 三个值的理论最大值


def _rmse_max_dis_len_angle(OD, r):
    # 1. 计算中心流
    centroid_flow = flow_centroid_OD(OD)
    # 2. 中心流长
    cl = flow_length(centroid_flow)
    return r, 2 * r, np.arcsin(2 * r / cl) if 2 * r <= cl else 0.00001


# 计算F是否为强连接簇，r是邻域
def _is_strong_flow_group_with_fixed_eps(OD: np.ndarray, eps: float) -> bool:
    AM, BM, CM = _rmse_max_dis_len_angle(OD, eps)
    A = flow_indicator_rmse_OD_distance(OD)
    B = flow_indicator_rmse_length(OD)
    C = flow_indicator_rmse_angle(OD)
    return A <= AM and B <= BM and C <= CM


def _is_strong_flow_group_with_fixed_eps_v2(OD: np.ndarray, eps: float) -> bool:
    # only use OD distance, not use length and angle
    return flow_indicator_rmse_OD_distance(OD) <= eps


def _scale_factor_func_linear(OD: np.ndarray, scale_factor: float) -> float:
    assert scale_factor >= 0 and scale_factor <= 1, f"Invalid scale_factor: {scale_factor}"
    return 0.5 * scale_factor * flow_length(OD)


def _scale_factor_func_sqrt(OD: np.ndarray, scale_factor: float) -> float:
    assert scale_factor >= 0 and scale_factor <= 1, f"Invalid scale_factor: {scale_factor}"
    return 0.5 * np.sqrt(scale_factor) * flow_length(OD)


def _scale_factor_func_square(OD: np.ndarray, scale_factor: float) -> float:
    assert scale_factor >= 0 and scale_factor <= 1, f"Invalid scale_factor: {scale_factor}"
    return 0.5 * np.square(scale_factor) * flow_length(OD)


def _scale_factor_func_tanh(OD: np.ndarray, scale_factor: float) -> float:
    assert scale_factor >= 0, f"Invalid scale_factor: {scale_factor}"
    return 0.5 * flow_length(OD) * (1 - np.exp(-scale_factor)) / (1 + np.exp(-scale_factor))


def _check_scale_factor_func(
    scale_factor_func: Union[Literal["linear", "square", "sqrt",
                                     "tanh"], Callable[[np.ndarray, float], np.ndarray]]
) -> Callable[[np.ndarray, float], np.ndarray]:
    if isinstance(scale_factor_func, Callable):
        pass

    elif isinstance(scale_factor_func, str):
        assert scale_factor_func in [
            "linear",
            "square",
            "sqrt",
            "tanh",
        ], f"Invalid scale_factor_func: {scale_factor_func}. Options: linear, square, sqrt, tanh, Please use one of them."
        scale_factor_func = eval(f"_scale_factor_func_{scale_factor_func}")

    else:
        raise RuntimeError(
            "Invalid scale_factor_func: {scale_factor_func}, must be callable or str, and str options: linear, square, sqrt, tanh.")

    return scale_factor_func


def _spatial_connected_flow_groups_can_discard(OD, k: int):
    return flow_number(OD) < k + 1


def _is_strong_connected_cluster_with_scale_factor(
    OD: np.ndarray,
    scale_factor: float = 0.1,
    scale_factor_func: Union[Literal["linear", "square", "sqrt", "tanh"], Callable[[
        np.ndarray, float], np.ndarray]] = _scale_factor_func_linear,
) -> bool:
    # 1. 计算中心流
    centroid_flow = flow_centroid_OD(OD)
    # 2. 中心流长
    f = _check_scale_factor_func(scale_factor_func)
    r = f(centroid_flow, scale_factor)
    return _is_strong_flow_group_with_fixed_eps(OD, r)


# 密度连接的过程
# 1. 首先找出所有核心流
# 2. 根据定义剔除部分核心流
# 3. 根据密度连接机制合并，更改标签
def flow_label_spatial_connected_groups_with_eps(OD_distance_matrix: np.ndarray, eps: Union[float, list, tuple, np.ndarray] = 0.5, min_flows: int = 5) -> np.ndarray[int]:
    lens = len(OD_distance_matrix)

    if hasattr(eps, '__len__'):
        if len(eps) < lens:
            eps = np.array([eps, [eps[-1]] * (lens - len(eps))]).flatten()
        else:
            eps = np.asarray(eps)
    else:
        eps = np.full(lens, fill_value=eps)

    assert min_flows > 0 and np.all(eps > 0), "Invalid params!"
    dm = OD_distance_matrix
    # 寻找核心流
    x = np.count_nonzero(dm <= eps, axis=1)
    core_flow_indices = np.where(x >= min_flows)[0]

    newdm = dm[core_flow_indices, :]
    y = np.count_nonzero(newdm <= eps, axis=1)
    core_flow_indices2 = np.where(y >= min_flows)[0]
    core_flow_indices = core_flow_indices[core_flow_indices2]

    label = np.full(lens, fill_value=-1)
    if len(core_flow_indices) == 0:
        return label

    # BFS遍历
    n = len(core_flow_indices)
    visited = np.full(n, False)
    groups = []

    for i, item in enumerate(core_flow_indices):
        if not visited[i]:
            group = []
            dq = deque()
            dq.append(item)
            visited[i] = True
            while dq:
                node = dq.popleft()
                group.append(node)
                for i2, item2 in enumerate(core_flow_indices):
                    if not visited[i2] and (dm[node, item2] <= eps[node] or dm[item2, node] <= eps[item2]):
                        dq.append(item2)
                        visited[i2] = True

            groups.append(np.asarray(group, dtype=int))

    for idx, gg in enumerate(groups):
        label[gg] = idx
    return label


# 这个版本引入尺度因子再计算得到密度连接簇
# 也就是每个r都等于0.5 * alpha * l
def flow_label_spatial_connected_groups_with_scale_factor(
    OD: np.ndarray,
    scale_factor: float = 0.1,
    min_flows: int = 5,
    flow_distance_func: Callable[[np.ndarray],
                                 np.ndarray] = flow_distance_matrix_max_euclidean,
    scale_factor_func: Union[Literal["linear", "square", "sqrt", "tanh"], Callable[[
        np.ndarray, float], np.ndarray]] = 'linear',
) -> np.ndarray[int]:
    f = _check_scale_factor_func(scale_factor_func)
    eps = f(OD, scale_factor)
    dm = flow_distance_func(OD)
    res = flow_label_spatial_connected_groups_with_eps(dm, eps, min_flows)
    del dm
    return res


# 计算FKI
def _flow_calculate_FKI(
    OD: np.ndarray, flow_distance_func: Callable[[np.ndarray], np.ndarray], k: int, calc_FKI_func: Callable[[np.ndarray], float], n_jobs: Optional[int] = None
) -> list[float]:
    distance_matrix = flow_distance_func(OD)

    nn = NearestNeighbors(n_neighbors=k, metric='precomputed', n_jobs=-1)
    nn.fit(distance_matrix)
    knn_idx = nn.kneighbors(return_distance=False)

    # seq = _flow_rearrange_arg(OD)
    if not n_jobs:
        KFI_res = []
        for i in range(flow_number(OD)):
            ll = knn_idx[i].tolist()
            ll.append(i)
            # get cluster
            sub_od = OD[ll, :]
            # calculate
            cur_indicator = calc_FKI_func(sub_od)
            KFI_res.append(cur_indicator)

        return KFI_res
    else:

        def process_batch(batch_indices):
            batch_results = []
            for i in batch_indices:
                ll = knn_idx[i].tolist()
                ll.append(i)
                sub_od = OD[ll, :]
                cur_indicator = calc_FKI_func(sub_od)
                batch_results.append(cur_indicator)
            return batch_results

        num_flows = flow_number(OD)
        indices = np.arange(num_flows)
        batch_size = 4
        batches = [indices[i: i + batch_size]
                   for i in range(0, num_flows, batch_size)]

        results = Parallel(n_jobs=n_jobs)(
            delayed(process_batch)(batch) for batch in batches)

        # Flatten the list of results
        KFI_res = [item for sublist in results for item in sublist]

        return KFI_res


# 请确保输入的FKI必须要是有序的
# 返回的数组要么等长，要么长度差一
# (FKI2 - FKI1) / FKI1
def _calculate_FKID(FKI: list) -> list[float]:
    FKI = np.asarray(FKI)
    if len(FKI) < 2:
        return []

    # 计算 (FKI[1:] - FKI[:-1]) / FKI[:-1]
    derivative = np.abs((FKI[1:] - FKI[:-1]) / FKI[:-1])

    return derivative


# 找到分割流的索引
def _flow_find_partitioningflow_arg(rearrange_arg: list, FKI: list, calc_FKID_func: Callable[[list], list]) -> int:
    if len(FKI) < 3:
        pf_idx = rearrange_arg[0]
    else:
        FKI = np.take(FKI, rearrange_arg)
        derivative = calc_FKID_func(FKI)
        if len(derivative) + 1 == len(FKI):
            pf_idx = rearrange_arg[np.argmax(derivative) + 1]
        elif len(derivative) == len(FKI):
            pf_idx = rearrange_arg[np.argmax(derivative)]
        else:
            raise RuntimeError(
                f"Check your calc_FKID_func! len(FKI): {len(FKI)}, len(FKID): {len(derivative)}")
    return pf_idx


# 返回左右子簇的索引
def _flow_get_left_right_indices_by_pf(rearrange_arg, pf_idx: int) -> tuple[list[int], list[int]]:
    assert pf_idx in rearrange_arg
    if not isinstance(rearrange_arg, list):
        rearrange_arg = list(rearrange_arg)
    idx = rearrange_arg.index(pf_idx)
    return rearrange_arg[:idx], rearrange_arg[idx + 1:]


def _flow_rearrange_indices(OD: np.ndarray) -> list[int]:
    if np.ndim(OD) == 1:
        return [0]

    # 1. get centroid flow
    cf = flow_centroid_OD(OD)
    cf_ang = flow_angle(cf)
    # 2. assign new O point
    OD = OD.copy()  # copy it
    OD[:, [0, 1]] = cf[:2]
    # 3. calculate all angles
    ang = flow_angle(OD)
    # 4. get reverse flow angle
    cf_rev_ang = np.mod(cf_ang + np.pi, 2 * np.pi)
    # 5. calculate the sequence
    all_ang = np.append(ang, cf_rev_ang)
    res = np.argsort(all_ang).tolist()  # it's a list
    cf_index = res.index(len(all_ang) - 1)

    return res[cf_index + 1:] + res[:cf_index]

# 返回分割后的左、右子簇的索引


def _flow_partition_arg(
    OD: np.ndarray,
    k: int,
    flow_distance_func: Callable[[np.ndarray], np.ndarray],
    calc_FKI_func: Callable[[np.ndarray], float],
    calc_FKID_func: Callable[[list], list],
    n_jobs: Optional[int] = None,
) -> tuple[list[int], list[int], int]:
    # 1. 重新排序
    new_idx = _flow_rearrange_indices(OD)
    # 2. 计算FKI
    FKI = _flow_calculate_FKI(OD, flow_distance_func,
                              k, calc_FKI_func, n_jobs=n_jobs)
    if 0 in FKI:
        FKI = [x + 1 for x in FKI]
        # _log.warning_once(
        #     "There are some FKI equals to 0, so we add 1 to them.")
    # 3. 计算寻找pf
    pf_idx = _flow_find_partitioningflow_arg(new_idx, FKI, calc_FKID_func)

    # 4. 获得左右的索引
    l, r = _flow_get_left_right_indices_by_pf(new_idx, pf_idx)
    return l, r, pf_idx
