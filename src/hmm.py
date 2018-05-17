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
from collections import defaultdict
from shapely.geometry import Point, LineString
from sklearn.model_selection import train_test_split

import tools

def get_obsv(piece_data):
    return piece_data[1][2:-2]

# 由于无第三方GPS，暂时使用paths作为train的mapmatching结果
def get_obsv_mat(train, match_res):
    D = dict()
    for tr_id, points in train.iteritems():
        matched = match_res[tr_id]
        for point in points:
            rid = matched[point[0]][3]
            obsv = get_obsv(point)
            if rid not in D.keys():
                D[rid] = dict()
            if obsv not in D[rid].keys():
                D[rid][obsv] = 0
            D[rid][obsv] += 1
    B = dict()
    for rid in D.iterkeys():
        total = sum(cnt for cnt in D[rid].itervalues())
        for obsv in D[rid].iterkeys():
            if not B.has_key(obsv):
                B[obsv] = dict()
            B[obsv][rid] = D[rid][obsv] * 1.0 / total
    return B, D

def get_trans_mat(match_res):
    A = dict()
    cnt = 0
    for tr_id, piece_data in match_res.iteritems():
        path = [x[3] for x in piece_data]
        for r1, r2 in tools.pairwise(path):
            if r1 not in A.keys():
                A[r1] = dict()
            if r2 not in A[r1].keys():
                A[r1][r2] = 0
            A[r1][r2] += 1
            cnt += 1
    print 'transition %d times' % cnt

    for r1 in A.iterkeys():
        total = sum(prob for prob in A[r1].itervalues()) * 1.0
        for r2 in A[r1].iterkeys():
            A[r1][r2] /= total
    return A

# given A, B, output state list R
def viterbi(A, B, traj):
    f, p = [], defaultdict(dict)
    h = dict()
    obsv = get_obsv(traj[0])
    for rid, prob in B[obsv].iteritems():
        h[rid] = prob
    f.append(h)
    for idx in range(1, len(traj)):
        obsv = get_obsv(traj[idx])
        h = dict()
        for rid, prob in B[obsv].iteritems():
            max_p = -0.1
            _obsv = get_obsv(traj[idx - 1])
            for _rid in B[_obsv].iterkeys():
                try:
                    alt_p = f[idx-1][_rid]+ A[_rid][rid] * B[obsv][rid]
                except:
                    alt_p = 0
                if alt_p > max_p:
                    max_p = alt_p
                    p[idx][rid] = _rid
                h[rid] = max_p
        f.append(h)
    sl = []
    max_prob = max(f[-1].itervalues())
    last_s = max(f[-1].iterkeys(), key=lambda k: f[-1][k])
    for idx in range(len(traj)-1, 0, -1):
        sl.append(last_s)
        last_s = p[idx][last_s]
    sl.append(last_s)
    sl.reverse()

    return sl, max_prob

def backforward(A, B, traj, matched, debug=False):
    traj_ = []
    states = []
    for point in traj:
        obsv = get_obsv(point)
        state = matched[point[0]][3]
        traj_.append((state, obsv))
    state_, obsv = traj_[0]
    prob = 1.0 * B[obsv][state_]
    states.append(state_)
    for idx in range(1, len(traj_)):
        state, obsv = traj_[idx]
        obsf = transf = 0
        try:
            obsf = B[obsv][state]
        except:
            if debug:
                print 'Obsv error', idx, obsv, state
        try:
            transf = A[state_][state]
        except:
            if debug:
                print 'Trans error', idx, state_, state
        prob += obsf * transf
        states.append(state)
        state_ = state
    return prob

def evaluate(test, predict, match_res):
    precision = dict()
    for tr_id, traj in test.iteritems():
        pdt = predict[tr_id]
        matched = match_res[tr_id]
        idx = 0
        error_cnt = 0
        for old_idx, point in traj:
            real_rid = matched[old_idx][3]
            pred_rid = pdt[idx]
            if real_rid != pred_rid:
                error_cnt += 1
            idx += 1
        precision[tr_id] = 1 - error_cnt * 1.0 / len(traj)
    return precision

def to_list(sl):
    last_rid, last_idx = 0, 0
    path = []
    for idx in range(len(sl)):
        rid = sl[idx]
        if rid!=last_rid and last_rid!=0:
            path.append((last_idx, idx-1, last_rid))
            last_idx = idx
        last_rid = rid
    path.append((last_idx, len(sl)-1, rid))
    return path
