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
import networkx as nx
import copy

class Sag:
    def __init__(self, side, bounding_box):
        self.side = side
        self.x0, self.x1, self.y0, self.y1 = bounding_box
        self.width = int(math.ceil((self.x1 - self.x0) / self.side))
        self.height = int(math.ceil((self.y1 - self.y0) / self.side))

    def utm2index(self, x, y):
        ix = int((x - self.x0) / self.side)
        iy = int((y - self.y0) / self.side)
        return ix, iy

    def utm_outside(self, x, y):
        return x < self.x0 or x > self.x1 or y < self.y0 or y > self.y1

    def utm2cell(self, x, y):
        if self.utm_outside(x, y):
            raise ValueError('utm outside')
        ix, iy = self.utm2index(x, y)
        return self.index2cell(ix, iy)

    def index2cell(self, ix, iy):
        return ix + iy * self.width

    def cell2index(self, cid):
        y = int(cid / self.width)
        x = cid % self.width
        return x, y

    def cell2utm(self, cid):
        x0, y0, x1, y1 = self.cell2box(cid)
        return (x0 + x1) / 2.0, (y0 + y1) / 2.0

    def cell2box(self, cid):
        x, y = self.cell2index(cid)
        x0, y0 = x * self.side + self.x0, y * self.side + self.y0
        return x0, y0, x0 + self.side, y0 + self.side

    def get_obsv(self, piece_data):
        return piece_data[1][2:-2]

    def get_cid(self, state):
        return state[0]

    def get_rid(self, state):
        return state[1]

    def cell_vec(self, c1, c2):
        x1, y1 = self.cell2index(c1)
        x2, y2 = self.cell2index(c2)
        return x2 - x1, y2 - y1

    def transfer(self, _state, state, more_info=False):
        prob = 1.0
        # path = []
        # if nx.has_path(self.graph, _state, state):
        #     path = nx.dijkstra_path(self.graph, _state, state)
        #     for c1, c2 in tools.pairwise(path):
        #         prob *= (1.0 - self.graph[c1][c2]['weight'])
        # else:
        #     prob = 0
        # if more_info:
        #     return prob, path
        # else:
        #     return prob
        return prob

    def get_trans_mat(self, match_res, match_ids):
        A = dict()
        passes = dict()
        for tr_id in match_ids:
            traj = match_res[tr_id]
            passed = []
            for idx, point in enumerate(traj):
                rid, mlat, mlng = point[3], point[4], point[5]
                x, y, _, _ = utm.from_latlon(mlat, mlng)
                try:
                    cid = self.utm2cell(x, y)
                except ValueError as err:
                    print tr_id, point
                    continue
                passed.append(cid)
            passes[tr_id] = passed
        for tr_id, passed in passes.iteritems():
            for c1, c2 in tools.pairwise(passed):
                if c1 not in A.keys():
                    A[c1] = dict()
                if c2 not in A[c1].keys():
                    A[c1][c2] = 0
                A[c1][c2] += 1
        for r1 in A.iterkeys():
            total = 0.0
            for r2 in A[r1].iterkeys():
                total += A[r1][r2]
            for r2 in A[r1].iterkeys():
                A[r1][r2] /= total
        self.graph = tools.construct_graph(A)
        return A, passes


    def get_obsv_mat(self, train, match_res, version=0):
        D = dict()
        for tr_id in train.iterkeys():
            traj = train[tr_id]
            matched = match_res[tr_id]
            for idx, point in enumerate(traj):
                piece_match = matched[point[0]]
                rid, lat, lng = piece_match[3], piece_match[4], piece_match[5]
                x, y, _, _ = utm.from_latlon(lat, lng)
                cid = self.utm2cell(x, y)
                obsv = self.get_obsv(point)
                # changed here!
                state = (cid, rid)
                if not D.has_key(state):
                    D[state] = dict()
                if not D[state].has_key(obsv):
                    D[state][obsv] = 0
                D[state][obsv] += 1
        B = dict()
        for state in D.iterkeys():
            total = sum(D[state].itervalues())
            for obsv in D[state].iterkeys():
                if not B.has_key(obsv):
                    B[obsv] = dict()
                B[obsv][state] = D[state][obsv] * 1.0 / total
        if version == 0:
            pass
        elif version == 1:
            for obsv in B.iterkeys():
                total = 0
                for state, prob in B[obsv].iteritems():
                    total += sum(D[state].itervalues())
                for state, prob in B[obsv].iteritems():
                    B[obsv][state] = D[state][obsv] * 1.0 / total
        elif version == 2:
            for obsv in B.iterkeys():
                total = 0
                for state, prob in B[obsv].iteritems():
                    total += sum(D[state].itervalues()) * D[state][obsv]
                for state, prob in B[obsv].iteritems():
                    B[obsv][state] = sum(D[state].itervalues()) * D[state][obsv] * 1.0 / total
        elif version == 3:
            for obsv in B.iterkeys():
                total = 0
                for state, prob in B[obsv].iteritems():
                    total += sum(D[state].itervalues()) * D[state][obsv]
                for state, prob in B[obsv].iteritems():
                    B[obsv][state] *= sum(D[state].itervalues()) * D[state][obsv] * 1.0 / total
        return B, D

    def get_cells(self, cell_db):
        cells = set()
        sp_dict = dict()
        for traj in cell_db.itervalues():
            for state in traj:
                cells.add(self.get_cid(state))
        for cid in cells:
            sp_dict[cid] = self.cell2box(cid)
        return sp_dict

    def viterbi(self, A, B, traj, k, debug=False):
        f, p = [], defaultdict(dict)
        h = dict()
        obsv = self.get_obsv(traj[0])
        for cid, prob in B[obsv].iteritems():
            h[cid] = prob
        f.append(h)
        for idx in range(1, len(traj)):
            obsv = self.get_obsv(traj[idx])
            speed, timestp = traj[idx][1][-2:]
            h = dict()
            _obsv = self.get_obsv(traj[idx-1])
            _speed, _timestp = traj[idx-1][1][-2:]
            max_dist = max(_speed, speed) * (timestp - _timestp) + 1.0 * self.side
            min_dist = min(_speed, speed) * (timestp - _timestp) - 1.0 * self.side
            for cid, prob in B[obsv].iteritems():
                max_p = 0
                x1, y1 = self.cell2index(cid)
                for _cid in B[_obsv].iterkeys():
                    if not f[idx-1].has_key(_cid):
                        continue
                    x2, y2 = self.cell2index(_cid)
                    avg_act_dist = (math.sqrt((x2-x1)**2 + (y2-y1)**2)) * self.side
                    t1 = t2 = t3 = 0
                    t1 = f[idx-1][_cid]
                    t3 = B[obsv][cid]
                    if not (min_dist - self.side * k <= avg_act_dist <= max_dist + self.side * k):
                        if debug:
                            print 'idx=%d,(%d,%d)'%(idx, _cid, cid)
                            print 'dist(%d,%d) act(%d)' % (int(min_dist), int(max_dist), int(avg_act_dist))
                        continue
                    t2 = self.transfer(_cid, cid)
                    if avg_act_dist < min_dist:
                        t2 *= math.e ** (-(avg_act_dist - min_dist)**2/(min_dist**2)/2)
                    if avg_act_dist > max_dist:
                        t2 *= math.e ** (-(avg_act_dist - max_dist)**2/(max_dist**2)/2)
                    alt_p = t1 + t2 * t3
                    if alt_p > max_p:
                        max_p = alt_p
                        p[idx][cid] = _cid
                    h[cid] = max_p
            f.append(h)
            if debug:
                print idx
        sl = []
        max_prob = max(f[-1].itervalues())
        last_s = max(f[-1].iterkeys(), key=lambda k: f[-1][k])
        for idx in range(len(traj)-1, 0, -1):
            sl.append(last_s)
            last_s = p[idx][last_s]
        sl.append(last_s)
        sl.reverse()
        
        return sl, max_prob

    def viterbi_2(self, A, B, traj, k, help_list, debug=False):
        f, p = [], defaultdict(dict)
        h = dict()
        obsv = self.get_obsv(traj[0])
        help_dict = help_list[0]
        for state, prob in B[obsv].iteritems():
            rid = self.get_rid(state)
            cid = self.get_cid(state)
            if help_dict.has_key(rid):
                h[cid] = prob * help_dict[rid]
        f.append(h)
        for idx in range(1, len(traj)):
            help_dict = help_list[idx]
            obsv = self.get_obsv(traj[idx])
            speed, timestp = traj[idx][1][-2:]
            h = dict()
            _obsv = self.get_obsv(traj[idx-1])
            _speed, _timestp = traj[idx-1][1][-2:]
            max_dist = max(_speed, speed) * (timestp - _timestp) + 1.0 * self.side
            min_dist = min(_speed, speed) * (timestp - _timestp) - 1.0 * self.side
            for state, prob in B[obsv].iteritems():
                max_p = 0
                rid = self.get_rid(state)
                cid = self.get_cid(state)
                if not help_dict.has_key(rid):
                    continue
                x1, y1 = self.cell2index(cid)
                for _state in B[_obsv].iterkeys():
                    _rid = self.get_rid(state)
                    _cid = self.get_cid(state)
                    if not f[idx-1].has_key(_cid):
                        continue
                    x2, y2 = self.cell2index(_cid)
                    avg_act_dist = (math.sqrt((x2-x1)**2 + (y2-y1)**2)) * self.side
                    t1 = t2 = t3 = 0
                    t1 = f[idx-1][_cid]
                    t3 = B[obsv][state]
                    if not (min_dist - self.side * k <= avg_act_dist <= max_dist + self.side * k):
                        if debug:
                            print 'idx=%d,(%d,%d)'%(idx, _cid, cid)
                            print 'dist(%d,%d) act(%d)' % (int(min_dist), int(max_dist), int(avg_act_dist))
                        continue
                    t2 = self.transfer(_cid, cid)
                    if avg_act_dist < min_dist:
                        t2 *= math.e ** (-(avg_act_dist - min_dist)**2/(min_dist**2)/2)
                    if avg_act_dist > max_dist:
                        t2 *= math.e ** (-(avg_act_dist - max_dist)**2/(max_dist**2)/2)
                    alt_p = t1 + t2 * t3
                    if alt_p > max_p:
                        max_p = alt_p
                        p[idx][cid] = _cid
                    h[cid] = max_p
            f.append(h)
            if debug:
                print idx
        sl = []
        max_prob = max(f[-1].itervalues())
        last_s = max(f[-1].iterkeys(), key=lambda k: f[-1][k])
        for idx in range(len(traj)-1, 0, -1):
            sl.append(last_s)
            last_s = p[idx][last_s]
        sl.append(last_s)
        sl.reverse()
        
        return sl, max_prob

    def backforward(self, A, B, traj, matched, k, debug=False):
        if debug:
            print 'debug'
        traj_ = []
        states = []
        for point in traj:
            lat, lng = matched[point[0]][4:6]
            obsv = self.get_obsv(point)
            speed, timestp = point[1][-2:]
            x, y, _, _ = utm.from_latlon(lat, lng)
            state = self.utm2cell(x, y)
            traj_.append((state, obsv, speed, timestp))
        # print traj_
        state_, obsv, _speed, _timestp = traj_[0]
        x1, y1 = self.cell2index(state_)
        prob = 1.0 * B[obsv][state_]
        states.append(state_)
        # print state_, prob
        for idx in range(1, len(traj_)):
            state, obsv, speed, timestp = traj_[idx]
            max_dist = max(_speed, speed) * (timestp - _timestp) + 1.0 * self.side
            min_dist = min(_speed, speed) * (timestp - _timestp) - 1.0 * self.side
            x2, y2 = self.cell2index(state)
            avg_act_dist = (math.sqrt((x2-x1)**2 + (y2-y1)**2)) * self.side
            
            transf = obsvf = 0
            transf = self.transfer(state_, state)
            obsvf = B[obsv][state]
            if not (min_dist - self.side*k <= avg_act_dist <= max_dist + self.side*k):
                if debug:
                    print 'idx=%d,(%d,%d)'%(idx, state_, state)
                    print '_speed=%s speed=%s gap=%s side=%s' % (_speed, speed, timestp - _timestp, self.side)
                    print 'dist(%d,%d) act(%d)' % (int(min_dist), int(max_dist), int(avg_act_dist))
                pass
            # if idx >= 2:
            #     state__ = state_
            #     cnt = -1
            #     for sidx in range(idx-1, 0, -1):
            #         state__ = states[cnt - 1]
            #         if state__ != state_:
            #             break
            #         cnt = cnt - 1
            #     x0, y0 = self.cell_vec(state__, state_)
            #     len0 = math.sqrt(x0**2+y0**2)
            #     x1, y1 = self.cell_vec(state_, state)
            #     len1 = math.sqrt(x1**2+y1**2)
            #     cos_theta = (x0 * x1 + y0 * y1) / len0 / len1 if len0 != 0 and len1 != 0 else 0
            #     if cos_theta == -1:
            #         print 'idx=%d,(%s,%s,%s)'%(idx, str(self.cell2index(state__)), str(self.cell2index(state_)), str(self.cell2index(state)))
            #         break
            if avg_act_dist < min_dist:
                transf *= math.e ** (-(avg_act_dist - min_dist)**2/(min_dist**2)/2)
            if avg_act_dist > max_dist:
                transf *= math.e ** (-(avg_act_dist - max_dist)**2/(max_dist**2)/2)
            prob += transf * obsvf
            # print state, prob
            state_ = state
            _speed = speed
            _timestp = timestp
            x1, y1 = x2, y2
            states.append(state_)
        return prob


    def evaluate(self, test, predict, match_res):
        precision = []
        precision_ = dict()
        self.utm_test, self.utm_pred = {}, {}
        self.cid_test, self.cid_pred = {}, {}
        self.gps_test, self.gps_pred = {}, {}
        for tr_id, pdt in predict.iteritems():
            traj = test[tr_id]
            matched = match_res[tr_id]
            avg = 0.0
            self.utm_test[tr_id], self.utm_pred[tr_id] = [], []
            self.cid_test[tr_id], self.cid_pred[tr_id] = [], []
            self.gps_test[tr_id], self.gps_pred[tr_id] = [], []
            for idx, point in enumerate(traj):
                # ground truth
                lat, lng = matched[point[0]][4:6]
                x, y, _, _ = utm.from_latlon(lat, lng)
                cid = pdt[idx]
                # prediction
                px, py = self.cell2utm(cid)

                cx, cy = self.utm2index(x, y)
                gx, gy = utm.to_latlon(px, py, 51, 'R')
                # for disp
                self.utm_test[tr_id].append((x, y))
                self.utm_pred[tr_id].append((px, py))
                self.gps_test[tr_id].append((lat, lng))
                self.gps_pred[tr_id].append((gx, gy))
                self.cid_test[tr_id].append((cx, cy))
                self.cid_pred[tr_id].append(self.cell2index(cid))
                dist = math.sqrt((px - x)**2 + (py - y)**2)
                precision.append(dist)
                avg += dist
            precision_[tr_id] = avg / len(traj)
        return precision, precision_


