""" 
Use flow L-Function to cluster flows.

Main steps of the algorithm:

1. Determine the best radius using global L-function derivative
2. Calculate the local K-Function under the best radius
3. Identify core flows Monte Carlo test
4. Cluster the core flows by connected mechanism

Paper Reference: Shu H. et al. (2021). “L-function of geographical flows.” International Journal of Geographical Information Science 35(4), 689–716.


How to use:
    from algorithm.FlowLF import flow_cluster_LF
    OD = ...
    # to specify the radius range and step size
    labels = flow_cluster_LF(OD, radius_low=1, radius_high=5, radius_step=0.1, significance=0.05, MonteCarloTest_times=199)
"""

from collections import deque
from sklearn.metrics import pairwise_distances
from algorithm.util_tools import *
from algorithm.ScaleFC import _scale_factor_func_linear
import numpy as np
from typing import Any, Callable, Generator, Iterable, Optional, Union
from joblib import Parallel, delayed
from sklearn.cluster import AgglomerativeClustering


def flow_cluster_LF(
    OD: np.ndarray,
    *,
    lambda_: Optional[float] = None,
    radius: Optional[float] = None,
    radius_low: Optional[float] = None,
    radius_high: Optional[float] = None,
    radius_step: Optional[float] = None,
    significance: float = 0.1,
    MonteCarloTest_times: int = 199,
    n_jobs: Optional[int] = None,
    link_by_radius: bool = True,
    min_num_of_subcluster: Optional[int] = 5,
    return_origin_label: bool = False,
    **kwargs,
) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """
    Calculates the flow cluster using the Ripley local L-Function.

    Paper Reference: Shu H. et al. (2021). “L-function of geographical flows.” International Journal of Geographical Information Science 35(4), 689–716.

    Args:
        OD (np.ndarray): The OD data, has 4 columns: ox, oy, dx, dy.
        lambda_ (float): The lambda value, if it's None, it will be calculated by the number and So x Sd.
        radius (Optional[float], optional): The radius value. Defaults to None. If this is None, use radius_low, radius_high, and radius_step to determine the best radius automatically.
        radius_low (Optional[float], optional): The lower bound of the radius. Required if radius is None. Defaults to None.
        radius_high (Optional[float], optional): The upper bound of the radius. Required if radius is None. Defaults to None.
        radius_step (Optional[float], optional): The step size for the radius. Required if radius is None. Defaults to None.
        significance (float, optional): The significance level. Defaults to 0.1.
        MonteCarloTest_times (int, optional): The number of Monte Carlo tests. Defaults to 199.
        n_jobs (Optional[int], optional): The number of jobs to run in parallel. Defaults to None. -1 means using all available cores.
        link_by_radius (bool, optional): Whether to link clusters by radius. Defaults to True.
        min_num_of_subcluster (int, optional): The minimum number of subclusters. Defaults to 5.
        return_origin_label (bool, optional): if True and link_by_radius is True and min_num_of_subcluster is not None, return the origin label(the label that hasn't been processed by min_num_of_subcluster), otherwise return the filtered label only. Defaults to False.
        **kwargs:
            flow_distance_func (Callable[[np.ndarray], np.ndarray], optional): The function to calculate the distance matrix. Defaults to flow_distance_matrix_max_euclidean.
            flow_distance_other_flow_func (Callable[[np.ndarray, np.ndarray], np.ndarray], optional): The function to calculate the distance matrix between two flows. Defaults to flow_distance_other_flow_matrix_max_euclidean.
            is_baised (bool, optional): Whether to use the biased estimator. Defaults to True.
            minus_one (bool, optional): Whether to subtract 1 from the result. Defaults to True.
            shuffle_OD_func (Callable[[np.ndarray, np.ndarray], Generator], optional): The function to shuffle the OD data. Defaults to flow_shuffle_from_OPoints_and_DPoints.

    Returns:
        np.ndarray: The cluster labels as a numpy array.
        tuple[np.ndarray, np.ndarray]: if return_origin_label and link_by_radius and min_num_of_subcluster, return (label, origin_label)
    """

    assert (MonteCarloTest_times +
            1) % 100 == 0, f"Invalid MonteCarloTest_times: {MonteCarloTest_times}"
    bound = int((MonteCarloTest_times + 1) * significance)
    assert bound % 2 == 0 and bound > 0, f"Invalid bound: {bound}"
    half_bound = bound // 2

    if (flow_distance_func := kwargs.get("flow_distance_func")) is None:
        flow_distance_func = flow_distance_matrix_max_euclidean

    if (flow_distance_other_flow_func := kwargs.get("flow_distance_other_flow_func")) is None:
        flow_distance_other_flow_func = flow_distance_other_flow_matrix_max_euclidean

    if (is_baised := kwargs.get("is_baised")) is None:
        is_baised = True

    if (minus_one := kwargs.get("minus_one")) is None:
        minus_one = True

    if (shuffle_OD_func := kwargs.get("shuffle_OD_func")) is None:
        shuffle_OD_func = flow_shuffle_from_OPoints_and_DPoints

    OD_distance_matrix = flow_distance_func(OD)
    if lambda_ is None:
        n = flow_number(OD)
        op, dp = flow_OD_points(OD)
        V = point_max_enclosing_circle_radius(
            op) * point_max_enclosing_circle_radius(dp)
        lambda_ = n / V
    if radius is None:
        assert radius_low is not None and radius_high is not None and radius_step is not None, "If radius is None, radius_low, radius_high, radius_step must be not None"
        radius = Ripley_HFunction_determine_best_radius(
            OD_distance_matrix, lambda_, is_baised, minus_one, radius_low, radius_high, radius_step, 2, 4)

    test_res = flow_Ripley_local_HFunction_MonteCarloTest(
        OD=OD,
        times=MonteCarloTest_times,
        lambda_=lambda_,
        radius=radius,
        is_baised=is_baised,
        pi_order=2,
        radius_order=4,
        shuffle_OD_func=shuffle_OD_func,
        distance_matrix_func=flow_distance_other_flow_func,
        n_jobs=n_jobs,
    )
    test_res = test_res.transpose((2, 1, 0))

    observed_res = Ripley_HFunction_local_base(
        OD_distance_matrix, lambda_, radius, is_baised, minus_one, 2, 4)
    observed_res = observed_res.transpose((1, 0))
    assert observed_res.shape[0] == test_res.shape[
        0], f"Invalid test_res: {test_res.shape}, observed_res: {observed_res.shape}"
    num_radius = test_res.shape[0]
    # Currently only supports one radius, not an array
    assert num_radius == 1, f"Invalid radius, is must be float or int: {test_res.shape}"
    num_OD = flow_number(OD)
    label = np.full(num_OD, -1)
    for j in range(num_OD):
        ob_value = observed_res[0, j]
        all_local_value = test_res[0, j, :]
        sv = np.sort(all_local_value)
        if ob_value > sv[-half_bound]:
            label[j] = 0

    if link_by_radius:
        idxx = label == 0
        # Connecting clusters based on the principle of minimum distance
        new_label = AgglomerativeClustering(n_clusters=None, linkage='single', metric="precomputed",
                                            distance_threshold=radius).fit_predict(OD_distance_matrix[idxx][:, idxx])
        label[idxx] = new_label
        if min_num_of_subcluster:
            origin_label = np.copy(label)
            label = clusters_relabel_by_number_of_each_subcluster(
                label, min_num_of_subcluster)
            if return_origin_label:
                return label, origin_label
        else:
            print("min_num_of_subcluster is None or 0, will generate subclusters with few flows.")

    return label


def flow_shuffle_from_OPoints_and_DPoints(OD: np.ndarray, times: int = -1) -> Generator:
    op, dp = flow_OD_points(OD)
    n_samples = flow_number(OD)
    idx = np.arange(n_samples)
    i = 0
    while i < times:
        i += 1
        np.random.shuffle(idx)
        new_op = op[idx, :]
        np.random.shuffle(idx)
        new_dp = dp[idx, :]
        res = np.hstack([new_op, new_dp])
        assert res.shape == (n_samples, 4)
        yield res


def point_max_enclosing_circle_radius(points: np.ndarray) -> float:
    from scipy.spatial import ConvexHull
    if points.shape[1] != 2:
        raise ValueError("Input points must be an N x 2 matrix")
    # Compute the convex hull
    hull = ConvexHull(points)
    # Get the vertices of the convex hull
    hull_points = points[hull.vertices]
    # Compute the pairwise distances between convex hull vertices
    distances = pairwise_distances(hull_points, hull_points, n_jobs=-1)
    # The radius of the maximum enclosing circle is half of the maximum distance
    max_radius = np.max(distances) / 2
    return max_radius


def Ripley_HFunction_local_base(
    distance_matrix: np.ndarray, lambda_: float, radius: Union[float, np.ndarray], is_baised: bool = True, minus_one: bool = True, pi_order: int = 1, radius_order: int = 2
) -> Union[float, np.ndarray]:
    lf_value = Ripley_LFunction_local_base(
        distance_matrix, lambda_, radius, is_baised, minus_one, pi_order, radius_order)
    return lf_value - radius


def Ripley_LFunction_local_base(
    distance_matrix: np.ndarray, lambda_: float, radius: Union[float, np.ndarray], is_baised: bool = True, minus_one: bool = True, pi_order: int = 1, radius_order: int = 2
) -> Union[float, np.ndarray]:
    kv = Ripley_KFunction_local_base(
        distance_matrix, lambda_, radius, is_baised, minus_one)
    return np.power(kv / np.power(np.pi, pi_order), 1 / radius_order)


def Ripley_KFunction_local_base(distance_matrix: np.ndarray, lambda_: float, radius: Union[float, int, np.ndarray], is_baised: bool = True, minus_one: bool = True) -> np.ndarray:
    assert np.ndim(
        distance_matrix) == 2, "Invalid distance similarity matrix: shape error."

    n = len(distance_matrix)
    # assert n == m, f"Invalid distance similarity matrix-shape: dim error."

    if not isinstance(radius, Iterable):
        radius = np.array([radius])

    min_dis = np.min(distance_matrix[distance_matrix != 0])
    assert np.all(
        radius >= min_dis), f"Invalid radius: all radius must >= min_dis in distance_matrix, radius: {radius}, min_dis: {min_dis}"

    res = np.zeros((n, len(radius)), dtype=int)
    for i, r in enumerate(radius):
        res[:, i] = np.sum(distance_matrix < r, axis=1)

    if minus_one:
        res = res - 1

    if is_baised:
        res = res / lambda_ / n
    else:
        res = res / lambda_ / (n - 1)

    return res


def Ripley_HFunction_determine_best_radius(
    distance_matrix: np.ndarray,
    lambda_: float,
    is_baised: bool = True,
    minus_one: bool = True,
    radius_low: float = 0.1,
    radius_high: float = 1,
    radius_step: float = 0.1,
    pi_order: int = 1,
    radius_order: int = 2,
) -> float:
    return _inner_Ripley_KLH_Function_determin_best_radius(
        distance_matrix,
        Ripley_HFunction_global_base,
        radius_low,
        radius_high,
        radius_step,
        lambda_=lambda_,
        is_baised=is_baised,
        minus_one=minus_one,
        pi_order=pi_order,
        radius_order=radius_order,
    )


def _inner_Ripley_KLH_Function_determin_best_radius(
    distance_matrix: np.ndarray, Ripley_global_func: Callable, radius_low: float = 0.1, radius_high: float = 1, radius_step: float = 0.1, **func_kwargs
) -> float:

    assert radius_low > 0, f"Invalid radius: low must > 0, low: {radius_low}"
    assert radius_low < radius_high, f"Invalid radius: low must < high, low: {radius_low}, high: {radius_high}"
    assert radius_step < (
        radius_high - radius_low), f"Invalid radius: step must < (high - low), step: {radius_step}, low: {radius_low}, high: {radius_high}"
    all_radius = np.arange(radius_low, radius_high, radius_step)
    ag = Ripley_global_func(distance_matrix=distance_matrix,
                            radius=all_radius, **func_kwargs)
    max_value = np.max(ag)
    max_arg = np.where(ag == max_value)[0]
    # _log.debug(f"max value: {max_value}, max arg: {max_arg}, ag: {ag}")
    if max_arg.size > 2:
        print(
            f"Multiple maximum value detected, including observed radius: {all_radius[max_arg]}")
    ii = max_arg[0]
    right_ag = ag[ii:]

    # The first local minimum to the right of the global maximum of the K/L/H function
    if len(right_ag) > 1:
        dif = np.diff(right_ag)
        # The first point where the derivative is positive; if none, then at the maximum point
        newii = np.where(dif > 0)[0]
        if newii.size > 0:
            newii = newii[0]
            ii += newii

    res = all_radius[ii]
    return res


def Ripley_HFunction_global_base(
    distance_matrix: np.ndarray, lambda_: float, radius: Union[float, np.ndarray], is_baised: bool = True, minus_one: bool = True, pi_order: int = 1, radius_order: int = 2
) -> Union[float, np.ndarray]:
    lf_value = Ripley_LFunction_global_base(
        distance_matrix, lambda_, radius, is_baised, minus_one, pi_order, radius_order)
    return lf_value - radius


def Ripley_LFunction_global_base(
    distance_matrix: np.ndarray, lambda_: float, radius: Union[float, np.ndarray], is_baised: bool = True, minus_one: bool = True, pi_order: int = 1, radius_order: int = 2
) -> Union[float, np.ndarray]:
    kv = Ripley_KFunction_global_base(
        distance_matrix, lambda_, radius, is_baised, minus_one)
    return np.power(kv / np.power(np.pi, pi_order), 1 / radius_order)


def Ripley_KFunction_global_base(
    distance_matrix: np.ndarray, lambda_: float, radius: Union[float, np.ndarray], is_baised: bool = True, minus_one: bool = True
) -> Union[float, np.ndarray]:
    res = Ripley_KFunction_local_base(
        distance_matrix=distance_matrix, lambda_=lambda_, radius=radius, is_baised=is_baised, minus_one=minus_one)

    if res.shape[1] == 1:
        return np.sum(res)
    return np.sum(res, axis=0)


# Obtain a matrix of dimensions times X data_number(data) X len(radius), using Monte Carlo simulation to estimate parameters
def _inner_Ripley_local_KLH_Function_MonteCarloTest(
    data: np.ndarray, n_jobs: int, times: int, shuffle_data_func: Callable, distance_matrix_func: Callable, Ripley_local_func: Callable, **Ripley_local_func_kwargs
):
    if not n_jobs:
        res_list = []
        for newData in shuffle_data_func(data, times):
            dm = distance_matrix_func(data, newData)
            res = Ripley_local_func(
                distance_matrix=dm, **Ripley_local_func_kwargs)
            del dm
            res_list.append(res)
        return np.asarray(res_list)
    else:
        assert isinstance(
            n_jobs, int), f"n_jobs must be an integer, but got {n_jobs}"

        def process_single():
            newData = next(shuffle_data_func(data, 1))
            dm = distance_matrix_func(data, newData)
            res = Ripley_local_func(
                distance_matrix=dm, **Ripley_local_func_kwargs)
            del dm
            return res

        batch_size = 16
        results = []
        with Parallel(n_jobs=n_jobs) as parallel:
            for i in range(0, times, batch_size):
                batch_end = min(i + batch_size, times)
                res_list = parallel(delayed(process_single)()
                                    for _ in range(i, batch_end))
                results.extend(res_list)

        return np.asarray(results)


def Ripley_local_HFunction_MonteCarloTest(
    data: Any,
    times: int,
    shuffle_data_func: Callable,
    distance_matrix_func: Callable,
    lambda_: float,
    radius: Union[float, np.ndarray],
    is_baised: bool = True,
    pi_order: int = 1,
    radius_order: int = 2,
    n_jobs: Optional[int] = None,
) -> np.ndarray:
    return _inner_Ripley_local_KLH_Function_MonteCarloTest(
        data,
        n_jobs,
        times,
        shuffle_data_func,
        distance_matrix_func,
        Ripley_HFunction_local_base,
        lambda_=lambda_,
        radius=radius,
        is_baised=is_baised,
        minus_one=False,
        pi_order=pi_order,
        radius_order=radius_order,
    )


def flow_Ripley_local_HFunction_MonteCarloTest(
    OD: np.ndarray,
    times: int,
    lambda_: float,
    radius: Union[float, int, np.ndarray],
    is_baised: bool = True,
    pi_order: int = 1,
    radius_order: int = 2,
    shuffle_OD_func: Callable[[np.ndarray, np.ndarray],
                              Generator] = flow_shuffle_from_OPoints_and_DPoints,
    distance_matrix_func: Callable[[np.ndarray, np.ndarray],
                                   np.ndarray] = flow_distance_other_flow_matrix_max_euclidean,
    n_jobs: Optional[int] = None,
) -> np.ndarray:
    return Ripley_local_HFunction_MonteCarloTest(
        data=OD,
        times=times,
        shuffle_data_func=shuffle_OD_func,
        distance_matrix_func=distance_matrix_func,
        lambda_=lambda_,
        radius=radius,
        is_baised=is_baised,
        pi_order=pi_order,
        radius_order=radius_order,
        n_jobs=n_jobs,
    )


# --------------------------------------------------------------------------

def flow_cluster_LF_with_scale_factor(OD: np.ndarray, scale_factor: float, significance: float = 0.1, MonteCarloTest_times: int = 199, n_jobs: Optional[int] = None, **kwargs):
    N = flow_number(OD)
    if (flow_distance_func := kwargs.get("flow_distance_func")) is None:
        flow_distance_func = flow_distance_matrix_max_euclidean

    if (flow_distance_other_flow_func := kwargs.get("flow_distance_other_flow_func")) is None:
        flow_distance_other_flow_func = flow_distance_other_flow_matrix_max_euclidean
    min_num_of_subcluster = kwargs.get("min_num_of_subcluster", 5)
    assert isinstance(min_num_of_subcluster, int) and min_num_of_subcluster > 0, f"min_num_of_subcluster must be greater than 0"
        
    distance_matrix = flow_distance_func(OD)
    eps_array = _scale_factor_func_linear(OD, scale_factor)
    # print(f"eps array is {eps_array}")
    # Here is the observed number of each neighbor
    x = np.count_nonzero(distance_matrix <= eps_array, axis=1)
    # assert len(x) == N and x.size == N
    # Parallel Execution of Monte Carlo Simulation Using joblib
    def monte_carlo_iteration():
        newOD = next(flow_shuffle_from_OPoints_and_DPoints(OD, 1))
        newdm = flow_distance_other_flow_func(OD, newOD)
        neweps = _scale_factor_func_linear(newOD, scale_factor)
        newx = np.count_nonzero(newdm <= neweps, axis=1)
        return (x >= newx).astype(int)
    results = Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(monte_carlo_iteration)()
        for _ in range(MonteCarloTest_times)
    )
    # Accumulate the results
    res = np.ones(N)
    for result in results:
        res += result
    
    # print(res)
    high =  (MonteCarloTest_times + 1) - ((MonteCarloTest_times + 1) * significance // 2)
    
    # Find the indices in res that are greater than or equal to high, I want the index numbers
    core_flow_indices = np.where(res >= high)[0]
    
    label = np.full(N, fill_value=-1)
    if len(core_flow_indices) == 0:
        return label

    # BFS
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
                    if not visited[i2] and (distance_matrix[node, item2] <= eps_array[node] or distance_matrix[item2, node] <= eps_array[item2]):
                        dq.append(item2)
                        visited[i2] = True

            groups.append(np.asarray(group, dtype=int))

    for idx, gg in enumerate(groups):
        label[gg] = idx
    label = clusters_relabel_by_number_of_each_subcluster(label, min_num_of_subcluster)
    return label
     
     
    
    