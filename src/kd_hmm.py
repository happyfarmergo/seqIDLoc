# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import utm
import math
import os
import sys
import time
import random
import copy
import tools
from collections import defaultdict

from rtree import index
from collections import deque


def axis_length(box, axis):
    return box[axis][1] - box[axis][0]

def find_new_mid(obsv, val, axis):
    for idx, x in enumerate(obsv):
        if x[axis] >= val:
            return idx
    return -1

def slide_cell(low, high, now, min_len):
    bottom, up = low, high
    assert high - low >= min_len
    if high - min_len <= now:
        bottom = high - min_len
    elif low + min_len >= now:
        up = low + min_len
    else:
        bottom, up = now - min_len / 2, now + min_len / 2
    return bottom, up

def kd_split(points, axis, out_box, min_gps=5, min_len=80, cut_len=40):
    g_idx = 0
    rtree_idx = index.Index()
    spatial_dict = dict()
    queue = deque()
    queue.append((points, axis, out_box))
    cnt = 0
    while len(queue) != 0:
        points, axis, out_box = queue.pop()
        print 'len:', len(points)
        if len(points) == 0:
            continue
        if axis_length(out_box, axis) <= min_len:
            if axis_length(out_box, int(not axis)) <= min_len:
                rtree_idx.insert(g_idx, (out_box[0][0], out_box[1][0], out_box[0][1], out_box[1][1]))
                spatial_dict[g_idx] = (out_box[0][0], out_box[1][0], out_box[0][1], out_box[1][1])
                g_idx += 1
                print g_idx
                continue
            else:
                axis = int(not axis)
        new_obsv = sorted(points, key=lambda t: t[axis])
        in_box = (new_obsv[0][axis], new_obsv[-1][axis])
        low_bound, high_bound = in_box[0] - out_box[axis][0], out_box[axis][1] - in_box[1]
        
        assert low_bound >=0
        assert high_bound >=0

        line = [False, False]
        if low_bound >=cut_len:
            line[0] = True
        if high_bound >=cut_len:
            line[1] = True

        if len(points) == 2:
            print 'axis=',axis, 'in_box:', in_box, 'out_box:', out_box
            print line
        if True not in line:
            if len(points) <= 3:
                median = sum(x[axis] for x in new_obsv) / float(len(new_obsv))
                mid = find_new_mid(new_obsv, median, axis)
            else:
                mid = len(new_obsv)/2
                median = new_obsv[mid][axis]
            out_box_1 = copy.deepcopy(out_box)
            out_box_1[axis][1] = median
            out_box_2 = copy.deepcopy(out_box)
            out_box_2[axis][0] = median
            queue.append((new_obsv[mid:], (axis + 1) % 2, out_box_2))
            queue.append((new_obsv[:mid], (axis + 1) % 2, out_box_1))
        else:
            low, high = out_box[axis]
            if line[0]:
                low = new_obsv[0][axis]
            if line[1]:
                high = new_obsv[-1][axis]
            out_box_3 = copy.deepcopy(out_box)
            #1 single point
            if low == high:
                low, high = slide_cell(out_box_3[axis][0], out_box_3[axis][1], low, min_len)
                print 'single point'
            out_box_3[axis][0] = low
            out_box_3[axis][1] = high
            if len(points) == 2:
                print 'updated bound:', low, '-', high
            queue.append((new_obsv, (axis + 1) % 2, out_box_3))
        cnt +=1
    return rtree_idx, spatial_dict, cnt


def utm2cell(x, y, rt_idx, sp_dict):
    cids = list(rt_idx.intersection((x, y, x, y)))
    if len(cids) == 1:
        return cids[0]
    for cid in cids:
        x_, y_ = sp_dict[cid][2:]
        if x_ == x or y_ == y:
            return cid

def get_trans_mat(dbs, r_idx, sp_dict):
    A = dict()
    cnt = 0
    for tr_id, traj in dbs.iteritems():
        for p1, p2 in tools.pairwise(traj):
            lat1, lng1 = p1[0], p1[1]
            lat2, lng2 = p2[0], p2[1]
            x0, y0, _, _ = utm.from_latlon(lat1, lng1)
            x1, y1, _, _ = utm.from_latlon(lat2, lng2)
            cid1 = utm2cell(x0, y0, r_idx, sp_dict)
            cid2 = utm2cell(x1, y1, r_idx, sp_dict)
            # print '[%d]->[%d]' % (cid1, cid2)
            if cid1 not in A.keys():
                A[cid1] = dict()
            if cid2 not in A[cid1].keys():
                A[cid1][cid2] = 0
            A[cid1][cid2] += 1
        cnt += 1
    # print 'transition %d times' % cnt
    # for r1, item in A.iteritems():
    #     if len(item) == 1:
    #         print '(road %d-> road %d) with %d times' % (r1, item.keys()[0], item[item.keys()[0]])

    for r1 in A.iterkeys():
        total = 0.0
        for r2 in A[r1].iterkeys():
            total += A[r1][r2]
    #     print time
        for r2 in A[r1].iterkeys():
            A[r1][r2] /= total
    return A