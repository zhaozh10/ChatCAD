import meshio
import numpy as np
import scipy.spatial
import torch
from Knee.Graphmaking.methods.edge import add_to_edges, sort_Edges


def makeMesh(data):
    edges_Ek = [[], []]
    edges_Et = [[], []]
    edges_Eb = [[], []]

    # for b in [1]:
    for b in range(data.num):
        n_v = [len(data.v_2d[b][_]) for _ in range(data.slice)]
        n_v = np.nonzero(n_v)
        st_idx, ed_slice = n_v[0].min(), n_v[0].max()

        for s in range(data.slice):
            if len(data.v_2d[b][s]) > 0:
                lines = link_Ek(data, b, s)
                edges_Ek = add_to_edges(lines, edges_Ek)
            if s + 1 < data.slice and len(data.v_2d[b][s]) * len(data.v_2d[b][s + 1]) > 0:
                lines = link_Et(data, b, s)
                edges_Et = add_to_edges(lines, edges_Et)
            if s == st_idx or s == ed_slice:
                lines = link_Eb(data, b, s)
                edges_Eb = add_to_edges(lines, edges_Eb)

    edges_Ek = sort_Edges(edges_Ek)
    edges_Et = sort_Edges(edges_Et)
    edges_Eb = sort_Edges(edges_Eb)
    edges = torch.cat([edges_Ek, edges_Et, edges_Eb], dim=1)
    faces = make_faces(edges, len(data.v_3d))

    data.mesh_edges = edges
    data.mesh_faces = faces
    data.mesh_vertices = data.v_3d
    # with open("test.pkl", "wb") as f:
    #     pickle.dump(edges, f)
    #     pickle.dump(data.v_3d, f)
    #     pickle.dump(faces, f)
    # saveMesh(data, faces)
    # exit()

    # # sort
    # edges_Es = sort_Edges(edges_Es)
    # edges_Ec = sort_Edges(edges_Ec)
    # edges_Ea = sort_Edges(edges_Ea)
    # data.edges = torch.cat([edges_Es, edges_Ec, edges_Ea], dim=1)
    # data.Es = edges_Es
    # data.Ec = edges_Ec
    # data.Ea = edges_Ea
    return


def link_Ek(data, b, s):
    idx = data.v_idx[b][s]
    # add edge along surface
    lines = [idx[:-1], idx[1:]]
    # make loop
    lines[0].append(idx[-1])
    lines[1].append(idx[0])
    # test
    # if len(data.v_idx[b][s]) > 3:
    #     k1, k2, k3 = 0, len(data.v_idx[b][s]) // 3, len(data.v_idx[b][s]) // 3 * 2
    #     d = find_direction(data.v_2d[b][s][k1], data.v_2d[b][s][k2], data.v_2d[b][s][k3])
    #     print(b, s, len(data.v_idx[b][s]), d)
    # exit()
    return lines


def link_Et(data, b, s):
    # make edge in adjacency slices
    lines = [[], []]

    # small set link to big set
    # print(len(data.v_2d[b][s]), len(data.v_2d[b][s + 1]))
    if len(data.v_2d[b][s]) > len(data.v_2d[b][s + 1]):
        vertex_b = data.v_2d[b][s]
        vertex_s = data.v_2d[b][s + 1]
        idx_b, idx_s = data.v_idx[b][s], data.v_idx[b][s + 1]
    else:
        vertex_b = data.v_2d[b][s + 1]
        vertex_s = data.v_2d[b][s]
        idx_b, idx_s = data.v_idx[b][s + 1], data.v_idx[b][s]
    n_s = len(vertex_s)
    n_b = len(vertex_b)
    # print(b, s, n_b, n_s)

    # concat the closest vertex
    mytree1 = scipy.spatial.cKDTree(vertex_b)
    kd_dist, clost_to_s = mytree1.query(vertex_s)
    # fix bugs
    # print(clost_to_s)
    min_idx = np.argmin(clost_to_s)
    for _i in range(n_s - 1):
        if clost_to_s[(min_idx + _i + 1) % n_s] < clost_to_s[(min_idx + _i) % n_s]:
            clost_to_s[(min_idx + _i + 1) % n_s] = clost_to_s[(min_idx + _i) % n_s]
    # print(clost_to_s)

    # exit()

    for sid, bid in enumerate(clost_to_s):
        # add line from big set to small set, each vertex in small set is connected
        lines[0].append(idx_s[sid])
        lines[1].append(idx_b[bid])
        # not all vertex in big set is connected, they connect ref by their neighbor
        while bid != clost_to_s[(sid + 1) % n_s]:
            bid = (bid + 1) % n_b
            lines[0].append(idx_s[sid])
            lines[1].append(idx_b[bid])

    if n_s == 1:
        # print(b, s)
        for bid in range(n_b):
            # add line from 1 point to big set, to ensure each vertex in big set is connected
            lines[0].append(idx_s[0])
            lines[1].append(idx_b[bid])

    # concat the closest vertex
    # mytree2 = scipy.spatial.cKDTree(vertex_s)
    # kd_dist, close_to_b = mytree2.query(vertex_b)
    # for bid, sid in enumerate(close_to_b):
    #     # add line from small set to big set, each vertex in big set is connected
    #     lines[0].append(idx_b[bid])
    #     lines[1].append(idx_s[sid])
    return lines


def link_Eb(data, b, s):
    idx = data.v_idx[b][s]
    n = len(idx)
    # add plane in the beside
    if n <= 3:
        return [[idx[0]], [idx[0]]]
    lines = [[], []]
    for _idx in range(n):
        lines[0].append(idx[0])
        lines[1].append(idx[_idx])
    return lines


def make_faces(edge, vertex_num):
    u = torch.cat((edge[0], edge[1]))
    v = torch.cat((edge[1], edge[0]))

    n = len(u)
    u_list = [[] for _ in range(vertex_num)]
    # print(u.max().item() + 1, "/", vertex_num)
    for i in range(n):
        u_list[u[i]].append(v[i].item())
    face = set()
    for p1 in range(vertex_num):
        for p2 in u_list[p1]:
            for p3 in u_list[p2]:
                if p1 in u_list[p3]:
                    f = [p1, p2, p3]
                    f.sort()
                    f = tuple(f)
                    face.add(f)
    face = list(face)
    # print("face num:", len(face))
    return face


def find_direction(p1, p2, p3):
    if ((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])) != 0:
        return ((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])) > 0
    else:
        print("point in a line.")
        exit()


def saveMesh(data, faces):
    mesh_name = "./test.obj"
    cells = [("triangle", np.array(faces))]
    mesh = meshio.Mesh(data.v_3d, cells)
    meshio.write(mesh_name, mesh)
