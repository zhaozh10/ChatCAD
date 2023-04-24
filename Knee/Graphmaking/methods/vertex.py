import Knee.Graphmaking.config as cfg
import meshio
import numpy as np
import scipy.spatial
import torch
from matplotlib.pyplot import axis


def extractVertex(data):
    data.v_2d = []
    data.v_3d = []
    data.v_idx = []
    data.thick = np.round(data.space[0] / data.space[2], 3)
    data.T_dist = (data.thick ** 2 + cfg.PATCH_SIZE ** 2) ** 0.5 * 0.8
    for b in range(data.num):
        xy = []
        sxy = []
        for s in range(data.slice):
            _xy = np.array(data.surface[b][s].nonzero(), dtype=np.float32)
            if len(_xy) > 0:
                xy_sorted = sort_xy(_xy)
                vertex = simplify_xy(xy_sorted)
                s_pc = np.zeros((vertex.shape[0], 1), dtype=np.float32) + s * data.thick
                sxy.append(np.hstack((s_pc, vertex)))
            else:
                vertex = []
            xy.append(vertex)
        data.v_2d.append(xy)
        data.v_3d.append(np.vstack(sxy))
    # plot_line(data.vertex[0][5])
    # plot_edge(data.vertex[0][9])
    idx = 0
    for b in range(data.num):
        data.v_idx.append([])
        for s in range(data.slice):
            data.v_idx[b].append([])
            for v in data.v_2d[b][s]:
                data.v_idx[b][s].append(idx)
                idx += 1
    data.v_3d = np.vstack(data.v_3d)
    data.v = data.v_3d
    data.v[:, 0] = np.round(data.v[:, 0] / data.thick, decimals=0).astype(np.int32)
    return


def sort_xy(points_list):

    _n = points_list.shape[0] - 1

    # find the first point that located at left top
    fisrt_idx = 0
    points_sort = [points_list[fisrt_idx]]
    points_list = np.delete(points_list, fisrt_idx, 0)

    # add point in sorted set and delete it from raw set
    for _i in range(_n):
        # find the point that closest to sorted set's first or latest points
        idx_l, idx_s = find_closest_point(points_list, points_sort)
        if idx_s == 0:
            points_sort.insert(0, points_list[idx_l])
        else:
            points_sort.append(points_list[idx_l])
        points_list = np.delete(points_list, idx_l, 0)
    points_sort = np.array(points_sort)

    # sort direction
    direction = False
    if len(points_sort) > 3:
        k1, k2, k3 = 0, len(points_sort) // 3, len(points_sort) // 3 * 2
        direction = find_direction(points_sort[k1], points_sort[k2], points_sort[k3])
    if direction:
        points_sort = np.flip(points_sort, axis=0)

    # # if edge is circled, make it be CCW
    # if cal_distance(points_sort[0], points_sort[-1]) < 8:
    #     for _i in range(_n + 1):
    #         if points_sort[_i][0] < points_sort[fisrt_idx][0]:
    #             fisrt_idx = _i
    #     if fisrt_idx != 0 and fisrt_idx != _n + 1:
    #         points_sort = np.vstack((points_sort[fisrt_idx:], points_sort[:fisrt_idx]))

    # xy = np.array(points_sort).transpose()
    # x = xy[0]
    # y = xy[1]
    # norm = plt.Normalize(y.min(), y.max())
    # norm_y = norm(y)
    # plt.figure()
    # plt.scatter(x, y, c=norm_y, cmap="viridis")
    # plt.savefig("./sorted_check_{}.png".format(idx))
    # exit()
    return np.array(points_sort)


def simplify_xy(points_sorted):

    # if the overlap between p_i and p_i+1 smaller than LAP_RATIO, add v_i+1 to vertex set
    _n = points_sorted.shape[0]
    vertex = []
    vertex.append(points_sorted[0])
    for _i in range(1, _n - 1):
        lap_ratio = cal_overlap(vertex[-1], points_sorted[_i])
        if lap_ratio < cfg.LAP_RATIO:
            vertex.append(points_sorted[_i])

    # add end point
    if cal_overlap(vertex[-1], points_sorted[-1]) > 0.5:
        vertex.pop()
    vertex.append(points_sorted[-1])

    if cal_overlap(vertex[0], vertex[-1]) > 0.5 and len(vertex) > 1:
        vertex.pop()

    if len(vertex) == 2:
        vertex.pop()
    # print(vertex)
    return np.array(vertex)


def find_closest_point(points_list, points_sort):
    points_sort_sne = [points_sort[0], points_sort[-1]]
    mytree1 = scipy.spatial.cKDTree(points_sort_sne)  # the sorted point set's start point and end point
    dist, idx = mytree1.query(points_list)
    min_idx = np.argmin(dist)
    return min_idx, idx[min_idx]


def cal_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def cal_distance3d(p1, p2, z_scale=1.0):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + z_scale * (p1[2] - p2[2]) ** 2) ** 0.5


def cal_overlap(p1, p2):
    w = cfg.PATCH_SIZE - np.abs(p1[0] - p2[0])
    h = cfg.PATCH_SIZE - np.abs(p1[1] - p2[1])
    if w < 0 or h < 0:
        return 0.0
    else:
        return w * h / cfg.PATCH_SIZE ** 2


def find_direction(p1, p2, p3):
    if ((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])) != 0:
        return ((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])) < 0
    else:
        return None
