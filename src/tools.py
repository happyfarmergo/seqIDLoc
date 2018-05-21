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