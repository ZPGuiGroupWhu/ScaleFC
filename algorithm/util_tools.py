from typing import Optional, Iterable
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors


######################################### FLOW ATTRIBUTES #################################

def flow_OD_points(OD, get_OPoints = True, get_DPoints = True):
    OD = np.reshape(OD, newshape=(-1, 4))
    if get_OPoints and get_DPoints:
        return OD[:, 0:2], OD[:, 2:4]
    elif get_OPoints:
        return OD[:, 0:2]
    elif get_DPoints:
        return OD[:, 2:4]
    return None


def flow_check_OD(OD, return2Darray=False):
    assert (np.ndim(OD) == 1 or np.ndim(OD) == 2) and ((np.ndim(OD) == 1 and np.shape(OD)[0] == 4) or np.shape(OD)[1] == 4), f"Invalid OD flow data, dim: {np.ndim(OD)}, OD: {OD}"
    if return2Darray:
        return np.reshape(OD, newshape=(-1, 4))


def flow_number(OD) -> int:
    # check the OD first
    flow_check_OD(OD)
    return np.size(OD) // 4

def flow_length(OD):
    OD = np.reshape(OD, newshape=(-1, 4))
    ox, oy, dx, dy = OD[:, 0], OD[:, 1], OD[:, 2], OD[:, 3]
    return np.sqrt(np.square(ox - dx) + np.square(oy - dy))


def flow2vector(OD) -> np.ndarray:
    if np.ndim(OD) == 1:
        ox, oy, dx, dy = OD[0], OD[1], OD[2], OD[3]
        return np.array([dx - ox, dy - oy])
    else:
        ox, oy, dx, dy = OD[:, 0], OD[:, 1], OD[:, 2], OD[:, 3]
    res = np.vstack((dx - ox, dy - oy)).T
    return res


def flow_angle(OD) :
    vec = flow2vector(OD).reshape(-1, 2)
    angles = np.arctan2(vec[:, 1], vec[:, 0])
    # 0-2π
    theta = np.mod(angles, 2 * np.pi)
    return theta

def flow_centroid_OD(OD):
    if np.ndim(OD) == 1:
        return np.asarray(OD)
    else:
        return np.mean(OD, axis=0)




######################################### FLOW DISTANCE #################################
# distance calculation
def _flow_O_points_dis_and_D_points_dis(OD1, metric = "euclidean", OD2= None):
    OD1 = np.reshape(OD1, (-1, 4))
    op = OD1[:, 0:2]
    dp = OD1[:, 2:4]
    if OD2 is None:
        op_dis = pairwise_distances(op, metric=metric, n_jobs=-1)
        dp_dis = pairwise_distances(dp, metric=metric, n_jobs=-1)
    else:
        OD2 = np.reshape(OD2, (-1, 4))
        op2 = OD2[:, 0:2]
        dp2 = OD2[:, 2:4]
        op_dis = pairwise_distances(op, op2, metric=metric, n_jobs=-1)
        dp_dis = pairwise_distances(dp, dp2, metric=metric, n_jobs=-1)
    return op_dis, dp_dis

# flow maximum distance
def flow_distance_other_flow_matrix_max_euclidean(OD1, OD2):
    op_dis, dp_dis = _flow_O_points_dis_and_D_points_dis(OD1, 'euclidean', OD2)
    res = np.maximum(op_dis, dp_dis)
    del op_dis, dp_dis
    return res

# flow maximum distance
def flow_distance_matrix_max_euclidean(OD):
    return flow_distance_other_flow_matrix_max_euclidean(OD, OD)


######################################### MISC FUNCTION #################################

def point_knn(points, k = 5, metric='euclidean'):
    nn = NearestNeighbors(n_neighbors=k, metric=metric, n_jobs=-1).fit(points)
    return nn.kneighbors(return_distance=False)


def point_knn_search_by_point(points, k= 5, metric='euclidean', search_points= None):
    nn = NearestNeighbors(n_neighbors=k, metric=metric, n_jobs=-1).fit(points)
    return nn.kneighbors(search_points, return_distance=False)


def clusters_relabel_by_number_of_each_subcluster(cluster_label, min_number):
    if np.all(cluster_label == -1):
        print("all lables in cluster_label is -1, please check your cluster_label.")
        return cluster_label
    assert min_number > 0, f"min_number must be greater than 0, but got {min_number}"

    v, c = np.unique(cluster_label, return_counts=True)
    if np.min(c[1:]) >= min_number:
        return cluster_label

    labels_res = []
    last_label = np.full(len(cluster_label), fill_value=-1, dtype=int)
    for val, cou in zip(v, c):
        if val == -1:
            continue
        if cou >= min_number:
            labels_res.append(np.where(cluster_label == val)[0])

    for idx, i in enumerate(labels_res):
        last_label[i] = idx

    return last_label


def flow_kth_nearest_neighbor_distance(OD: np.ndarray, k: int = 5, return_index: bool = False, flow_distance_func = flow_distance_matrix_max_euclidean):
    dm = flow_distance_func(OD)
    return kth_nearest_neighbor_distance(dm, k, return_index)


def kth_nearest_neighbor_distance(distance_matrix: np.ndarray, k: int = 5, return_index: bool = False) -> np.ndarray:
    if isinstance(k, int):
        use_k = k
    elif isinstance(k, Iterable):
        k = np.asarray(k, dtype=int)
        use_k = np.max(k)
    else:
        raise TypeError("k must be an integer or an iterable of integers.")
    nn = NearestNeighbors(n_neighbors=use_k, metric="precomputed", algorithm='auto', n_jobs=-1).fit(distance_matrix)
    # 返回的距离是排序的第k个
    dis, idx = nn.kneighbors()
    sls = (slice(None), k - 1)
    # print(return_index)
    return (dis[sls], idx[sls]) if return_index else dis[sls]