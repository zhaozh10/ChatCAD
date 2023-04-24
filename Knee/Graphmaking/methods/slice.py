import os
import pickle
from os.path import join

import config as cfg
import dgl
import numpy as np
import scipy.spatial
import torch

from methods.edge import add_to_edges, sort_Edges
from methods.patch import extract_patch, paintingPatch
from methods.vertex import cal_distance, simplify_xy, sort_xy


def extractSliceVertex(data):
    data.v_2d = []
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
    for s in range(data.slice):
        idx = 0
        for b in range(data.num):
            data.v_idx.append([])
            data.v_idx[b].append([])
            for v in data.v_2d[b][s]:
                data.v_idx[b][s].append(idx)
                idx += 1
    return


def extractSliceEdges(data):
    data.edges = []
    # make edge in cartilage (Es), edge between cartilage (Ec)
    for s in range(data.slice):
        edges_Es = [[], []]
        edges_Ec = [[], []]
        for b in range(data.num):
            lines = [[], []]
            if len(data.v_2d[b][s]) > 0:
                lines = link_Es(data, b, s)
                edges_Es = add_to_edges(lines, edges_Es)
        if data.num > 1:
            lines = link_Ec(data, s, data.num)
            edges_Ec = add_to_edges(lines, edges_Ec)
        # sort
        edges_Es = sort_Edges(edges_Es)
        edges_Ec = sort_Edges(edges_Ec)
        data.edges.append(torch.cat([edges_Es, edges_Ec], dim=1))
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


def extractSlicePatch(data):
    data.patch, data.bones, data.label, data.pos, data.graph = [], [], [], [], []
    mask = torch.zeros(data.mri.shape, dtype=bool)
    for s in range(data.slice):
        ptchs, bones, lesns, label, poses = [], [], [], [], []
        if len(data.edges[s]) > 0:
            for b in range(data.num):
                if len(data.v_2d[b][s]) > 0:
                    _ptchs = extract_patch(data.mri[s], data.v_2d[b][s])
                    _bones = extract_patch(data.bone[b][s], data.v_2d[b][s])
                    _lesns = extract_patch(data.les[b][s], data.v_2d[b][s])
                    _label = torch.max(_lesns.view(_lesns.shape[0], -1), dim=1).values
                    _pos = torch.tensor(data.v_2d[b][s])
                    _pos[:, 0] /= data.shape[0]
                    _pos[:, 1] /= data.shape[1]
                    ptchs.append(_ptchs)
                    bones.append(_bones)
                    lesns.append(_lesns)
                    label.append(_label)
                    poses.append(_pos)
                    mask[s] += paintingPatch(data.mri[s], data.v_2d[b][s])
            graph = dgl.graph((data.edges[s][0], data.edges[s][1]))
            data.patch.append(torch.cat(ptchs, dim=0).type(torch.float32))
            data.bones.append(torch.cat(bones, dim=0).type(torch.long))
            data.label.append(torch.cat(label, dim=0).type(torch.long))
            data.pos.append(torch.cat(poses, dim=0).type(torch.float32))
            data.graph.append(dgl.add_self_loop(graph))
    data.mask = mask


def saveSliceData(data):
    path = [join(cfg.OUTPUT, "graph"), join(cfg.OUTPUT, "vertex")]
    for _path in path:
        if not os.path.exists(_path):
            os.makedirs(_path)

    n = len(data.patch)
    for i in range(n):
        with open(join(path[0], f"{data.name}_{i}.npz"), "wb") as f:
            pickle.dump(data.graph[i], f)

        np.savez_compressed(join(path[1], f"{data.name}_{i}.npz"), patch=data.patch[i], bones=data.bones[i], label=data.label[i], pos=data.pos[i])

        if data.graph[i].num_nodes() != data.patch[i].shape[0]:
            print(f"Error in {data.idx}_{i}, vertex: {data.graph[i].num_nodes()}, patch: {data.patch[i].shape[0]}")
            exit()
        else:
            print(data.name, i, data.graph[i].num_nodes(), data.graph[i].num_edges(), np.round(data.space[0], 2), data.slice, data.rawshape, data.label[i].max().item(), cfg.OUTPUT, sep="|")

    return
