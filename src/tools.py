# -*- coding:utf-8 -*-
import networkx as nx
import utm
import math
from shapely.geometry import Point, LineString

def pairwise(iterable):
    import itertools
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return itertools.izip(a, b)

def outside(x, y, bounding_box):
    x0, x1, y0, y1 = bounding_box
    return x < x0 or x > x1 or y < y0 or y > y1

def construct_graph(A):
    graph = nx.DiGraph()
    for c1 in A.iterkeys():
        for c2 in A[c1].iterkeys():
            graph.add_edge(c1, c2, weight=1.0-A[c1][c2])
    return graph

def make_map(grid, roadmap, cut_length, step_len):
    # big_id: [cid,...]
    big_id = 0
    slice_map = dict()
    for rid, locs in roadmap.iteritems():
        coors = []
        outside = False
        for lat, lng in locs:
            x, y, _, _ = utm.from_latlon(lat, lng)
            if grid.utm_outside(x, y):
#                 print 'road id=%d, outside' % rid
                outside=True
                break
            coors.append((x, y))
        if outside:
            continue
        slice_map[rid] = dict()
        line_road = LineString(coors)
        line_length = line_road.length
        line_count = int(math.ceil(line_length / float(cut_length)))
#         print line_length, line_count
        turn_info = []
        for idx, coor in enumerate(coors[1:-1]):
            dist = line_road.project(Point(coor))
            turn_info.append((idx + 1, dist))
#         print turn_info
        # 将一条路划分为几个片段
        road_slices = []
        for i in range(line_count):
            cut_dist = cut_length * i
            cut_point = line_road.interpolate(cut_dist + cut_length)
            pre_cut_point = line_road.interpolate(cut_dist)
            slices = [(pre_cut_point.x, pre_cut_point.y)]
            temp = []
            for idx, dist in turn_info:
                if cut_dist < dist < cut_dist + cut_length:
                    slices.append(coors[idx])
                    temp.append(idx)
#             print i, temp
            slices.append((cut_point.x, cut_point.y))
#             print slices
            road_slices.append(slices)
        # 确定每个片段经过的格子
        last_cids = []
        for idx, slices in enumerate(road_slices):
            line_slice = LineString(slices)
#             print line_slice
            step_count = int(math.ceil(line_slice.length / float(step_len))) + 1 if idx == len(road_slices) - 1 else cut_length / step_len
#             print step_count
            cids = []
            last_cid = 0
            for i in range(step_count):
                cut_point = line_slice.interpolate(i*step_len)
#                 print cut_point.x, cut_point.y
                cid = grid.utm2cell(cut_point.x, cut_point.y)
#                 print grid.cell2index(cid)
                if cid != last_cid and cid not in last_cids:
                    cids.append(cid)
                last_cid = cid
            slice_map[rid][big_id] = cids
            big_id += 1
            last_cids = cids
    return slice_map

def cut_map_as_cells(grid, roadmap, step_len=1):
    slice_map = dict()
    for rid, locs in roadmap.iteritems():
        coors = []
        outside = False
        for lat, lng in locs:
            x, y, _, _ = utm.from_latlon(lat, lng)
            if grid.utm_outside(x, y):
#                 print 'road id=%d, outside' % rid
                outside=True
                break
            coors.append((x, y))
        if outside:
            continue
        line_road = LineString(coors)
        line_length = line_road.length
        line_count = int(math.ceil(line_length / float(step_len)))

        last_cid = -1
        cids = []
        for i in range(line_count + 1):
            cut_dist = step_len * i
            cut_point = line_road.interpolate(cut_dist)
            cid = grid.utm2cell(cut_point.x, cut_point.y)
            if cid != last_cid:
                cids.append(cid)
            last_cid = cid
        slice_map[rid] = cids
        # if len(cids) == 1:
        #     print rid, line_length, coors
    return slice_map

def find_roads(cid, dict_map, n_rid):
    inter_roads = []
    for rid, cids in dict_map.iteritems():
        if rid!=n_rid and cid in cids:
            inter_roads.append(rid)
    return inter_roads


def make_road_graph(grid, slice_map, price_map, version=0):
    if version == 0:
        return make_road_graph_1(grid, slice_map, price_map)
    elif version == 1:
        return make_road_graph_2(grid, slice_map, price_map)


def make_road_graph_1(grid, slice_map, price_map):
    dict_map = dict()
    graph = dict()
    dict_rid = dict()
    segid = 0
    for rid, cids in slice_map.iteritems():
        dict_map[rid] = set(cids)
    isol_rids = []
    for rid, cids in slice_map.iteritems():
        s_idx, e_idx = 0, 0
        rid_inters = []
        idxes = []
        for idx, cid in enumerate(cids):
            inter_rids = find_roads(cid, dict_map, rid)
            if len(inter_rids)!=0:
                rid_inters.extend(inter_rids)
                idxes.append(idx)
        if len(idxes) == 0:
            isol_rids.append(rid)
            continue

        graph[rid] = [cids, rid_inters, price_map[rid]]
    print 'isolated rids:', isol_rids
    weights = dict()
    for rid, data in graph.iteritems():
        weights[rid] = data[-1]
    return graph, weights


def make_road_graph_2(grid, slice_map, price_map):
    dict_map = dict()
    graph = dict()
    dict_rid = dict()
    segid = 0
    for rid, cids in slice_map.iteritems():
        dict_map[rid] = set(cids)
    isol_rids = []
    for rid, cids in slice_map.iteritems():
        s_idx, e_idx = 0, 0
        rid_inters = dict()
        idxes = []
        for idx, cid in enumerate(cids):
            inter_rids = find_roads(cid, dict_map, rid)
            if len(inter_rids)!=0:
                rid_inters[idx] = inter_rids
                idxes.append(idx)
        if len(idxes) == 0:
            isol_rids.append(rid)
            continue
        if idxes[0] != 0:
            idxes.insert(0,0)
            rid_inters[0] = []
        if idxes[-1] != len(cids) - 1:
            idxes.append(len(cids) - 1)
            rid_inters[len(cids)-1] = []

        assert idxes[0]==0 and idxes[-1] == len(cids) - 1

        segids = []
        for s_idx, e_idx in pairwise(idxes):
            segids.append(segid)
            data = [cids[s_idx:e_idx+1], rid_inters[s_idx], rid_inters[e_idx], price_map[rid] * (e_idx - s_idx + 1) / float(len(cids))]
            graph[segid] = data
            segid += 1
        dict_rid[rid] = segids
    print 'isolated rids:', isol_rids
    weights = dict()
    for segid, data in graph.iteritems():
        s_cid, e_cid = data[0][0], data[0][-1]
        neighbors = []
        for rid in data[1]:
            for s_segid in dict_rid[rid]:
                if s_cid in graph[s_segid][0]:
                    neighbors.append(s_segid)
        for rid in data[2]:
            for s_segid in dict_rid[rid]:
                if e_cid in graph[s_segid][0]:
                    neighbors.append(s_segid)
        graph[segid] = [data[0], neighbors]
        weights[segid] = data[-1]
    return graph, weights


def dijkstra(graph, weights, sid, eid):
    visited = {sid:weights[sid]}
    path = {}
    nodes = set(graph.keys())

    if sid == eid:
        return visited, path

    while nodes:
        min_node = None
        for node in nodes:
            if node in visited:
                if min_node is None:
                    min_node = node
                elif visited[node] < visited[min_node]:
                    min_node = node
        if min_node is None:
            break
        nodes.remove(min_node)
        cur_cost = visited[min_node]

        for next_node in graph[min_node][1]:
            cost = cur_cost + weights[next_node]
            if next_node not in visited or cost < visited[next_node]:
                visited[next_node] = cost
                path[next_node] = min_node
    return visited, path

