import Knee.Graphmaking.config as cfg
import numpy as np
import scipy.spatial
import torch
from Knee.Graphmaking.methods.vertex import cal_distance


def extractEdges(data):
    edges_Es = [[], []]
    edges_Ec = [[], []]
    edges_Ea = [[], []]

    # make edge in cartilage (Es), edge between cartilage (Ec) and edge betewwn slice (Ea)
    for b in range(data.num):
        for s in range(data.slice):
            if len(data.v_2d[b][s]) > 0:
                lines = link_Es(data, b, s)
                edges_Es = add_to_edges(lines, edges_Es)
            if data.num > 1:
                lines = link_Ec(data, s, data.num)
                edges_Ec = add_to_edges(lines, edges_Ec)
            if s + 1 < data.slice and len(data.v_2d[b][s]) * len(data.v_2d[b][s + 1]) > 0:
                lines = link_Ea(data, b, s)
                edges_Ea = add_to_edges(lines, edges_Ea)

    # sort
    edges_Es = sort_Edges(edges_Es)
    edges_Ec = sort_Edges(edges_Ec)
    edges_Ea = sort_Edges(edges_Ea)
    data.edges = torch.cat([edges_Es, edges_Ec, edges_Ea], dim=1)

    data.Es = edges_Es
    data.Ec = edges_Ec
    data.Ea = edges_Ea
    return


def link_Es(data, b, s):
    vertex = data.v_2d[b][s]
    idx = data.v_idx[b][s]
    # add edge along surface
    lines = [idx[:-1], idx[1:]]
    # add addation edge if it is a loop
    if cal_distance(vertex[0], vertex[-1]) < cfg.PATCH_SIZE * 1.4 * 0.8:
        lines[0].append(idx[-1])
        lines[1].append(idx[0])
    if b == 2:  # patella
        lines[0].append(idx[-1])
        lines[1].append(idx[0])
    return lines


def link_Ec(data, s, num):
    lines = [[], []]
    # add edge between cartilage
    for i in range(num):
        v_fix = data.v_2d[i][s]
        v_query = data.v_2d[(i + 1) % num][s]
        if len(v_fix) * len(v_query) > 0:
            for _idx, _v in enumerate(v_fix):
                mytree1 = scipy.spatial.cKDTree([_v])
                kd_dist, kd_idx = mytree1.query(v_query)
                for j in range(len(kd_dist)):
                    if kd_dist[j] < cfg.PATCH_SIZE * 1.4 * 0.35:
                        lines[0].append(data.v_idx[i][s][_idx])
                        lines[1].append(data.v_idx[(i + 1) % num][s][j])
    return lines


def link_Ea(data, b, s):
    lines = [[], []]
    # make edge in adjacency slices (Ea)
    vertex0 = data.v_2d[b][s]
    vertex1 = data.v_2d[b][s + 1]
    # concat the closest vertex
    for _idx, _v in enumerate(vertex0):
        mytree1 = scipy.spatial.cKDTree([_v])
        kd_dist, kd_idx = mytree1.query(vertex1)
        for j in range(len(kd_dist)):
            if kd_dist[j] < data.T_dist:
                lines[0].append(data.v_idx[b][s][_idx])
                lines[1].append(data.v_idx[b][s + 1][j])
    for _idx, _v in enumerate(vertex1):
        mytree1 = scipy.spatial.cKDTree([_v])
        kd_dist, kd_idx = mytree1.query(vertex0)
        for j in range(len(kd_dist)):
            if kd_dist[j] < data.T_dist:
                lines[0].append(data.v_idx[b][s + 1][_idx])
                lines[1].append(data.v_idx[b][s][j])
    return lines


def add_to_edges(lines, edges):
    edges[0] += lines[0]
    edges[1] += lines[1]
    return edges


def sort_Edges(edges):
    edges = np.array(edges).transpose().tolist()
    for edge in edges:
        edge.sort()
    edges.sort()
    _edges = []
    for e in edges:
        if e not in _edges:
            _edges.append(e)
    edges = _edges
    edges = torch.tensor(np.array(edges).transpose(), dtype=torch.int32)
    return edges


# def extractEdges(data):
# make_graph_edge(data)
# make_visualized_edge(data)
# make_mesh(data, data.graph_edge)
# make_mesh(data, data.visualized_edge)
# exit()
