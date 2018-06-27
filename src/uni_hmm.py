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
    def __init__(self, grid, slice_map, state_type):
        self.grid = grid
        self.map = slice_map
        self.state_type = state_type

    def get_obsv(self, piece_data):
        return piece_data[1][2:-2]

    def get_big_id(self, rid, cid):
        for big_id, cells in self.map[rid].iteritems():
            if cid in cells:
                return big_id
        return -1

    def uzip(self, state):
        return state[0], state[1]

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

    # def get_trans_mat(self, match_res):
    #     A = dict()
    #     passes = dict()
    #     for tr_id in match_res.iterkeys():
    #         traj = match_res[tr_id]
    #         passed = []
    #         for idx, point in enumerate(traj):
    #             rid, mlat, mlng = point[3], point[4], point[5]
    #             x, y, _, _ = utm.from_latlon(mlat, mlng)
    #             try:
    #                 cid = self.grid.utm2cell(x, y)
    #             except ValueError as err:
    #                 print tr_id, point
    #                 continue
    #             passed.append(cid)
    #         passes[tr_id] = passed
    #     for tr_id, passed in passes.iteritems():
    #         for c1, c2 in tools.pairwise(passed):
    #             if c1 not in A.keys():
    #                 A[c1] = dict()
    #             if c2 not in A[c1].keys():
    #                 A[c1][c2] = 0
    #             A[c1][c2] += 1
    #     for r1 in A.iterkeys():
    #         total = 0.0
    #         for r2 in A[r1].iterkeys():
    #             total += A[r1][r2]
    #         for r2 in A[r1].iterkeys():
    #             A[r1][r2] /= total
    #     # self.graph = tools.construct_graph(A)
    #     return A, passes

    def get_trans_mat(self, match_res):
        A = dict()
        passes = dict()
        for tr_id in match_res.iterkeys():
            traj = match_res[tr_id]
            passed = []
            for idx, point in enumerate(traj):
                timestp, rid, mlat, mlng = point[0], point[3], point[4], point[5]
                x, y, _, _ = utm.from_latlon(mlat, mlng)
                try:
                    cid = self.grid.utm2cell(x, y)
                except ValueError as err:
                    print tr_id, point
                    continue
                passed.append((cid, timestp))
            passes[tr_id] = passed
        for tr_id, passed in passes.iteritems():
            for step in range(1, len(passed)):
                for head in range(len(passed)-step):
                    c1, t1 = passed[head]
                    c2, t2 = passed[head + step]
                    d_t = t2 - t1
                    if d_t not in A.keys():
                        A[d_t] = dict()
                    if c1 not in A[d_t].keys():
                        A[d_t][c1] = dict()
                    if c2 not in A[d_t][c1].keys():
                        A[d_t][c1][c2] = 0
                    A[d_t][c1][c2] += 1
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
                cid = self.grid.utm2cell(x, y)
                obsv = self.get_obsv(point)
                # changed here!
                if self.state_type == 0:
                    state = cid
                elif self.state_type == 1:
                    big_id = self.get_big_id(rid, cid)
                    if big_id == -1:
                        print tr_id, idx, rid, cid
                    state = (cid, big_id)
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

    def get_cells(self, cells):
        sp_dict = dict()
        for state in cells:
            if self.state_type == 0:
                cid = state
            elif self.state_type == 1:
                cid, big_id = self.uzip(state)
            sp_dict[cid] = self.grid.cell2box(cid)
        return sp_dict

    def valid_trans(self, A, dt_1, dt_2, _cid, cid):
        for dt in range(dt_1, dt_2 + 1):
            if dt not in A.keys() or _cid not in A[dt].keys():
                continue
            if cid in A[dt][_cid].keys():
                return True
        return False

    def transfer(self, A, dt_1, dt_2, _cid, cid):
        total = 0
        cnt = 0
        for dt in range(dt_1, dt_2 + 1):
            if dt not in A.keys() or _cid not in A[dt].keys():
                continue
            total += sum(num for k, num in A[dt][_cid].iteritems())
            if cid in A[dt][_cid].keys():
                cnt += A[dt][_cid][cid]
        return cnt / float(total)

    def viterbi(self, A, B, traj, k, debug=False):
        f, p = [], defaultdict(dict)
        h = dict()
        obsv = self.get_obsv(traj[0])
        for state, prob in B[obsv].iteritems():
            h[state] = prob
        f.append(h)
        for idx in range(1, len(traj)):
            obsv = self.get_obsv(traj[idx])
            speed, timestp = traj[idx][1][-2:]
            h = dict()
            _obsv = self.get_obsv(traj[idx-1])
            _speed, _timestp = traj[idx-1][1][-2:]
            delta_t = timestp - _timestp
            sigma = 0.2
            dt_1, dt_2 = (1-sigma)*delta_t, (1+sigma)*delta_t
            max_dist = max(_speed, speed) * (timestp - _timestp) + 1.0 * self.grid.side
            min_dist = min(_speed, speed) * (timestp - _timestp) - 1.0 * self.grid.side
            for state, prob in B[obsv].iteritems():
                max_p = 0
                cid = state
                x1, y1 = self.grid.cell2index(cid)
                for _state in B[_obsv].iterkeys():
                    if not f[idx-1].has_key(_state):
                        continue
                    _cid = _state
                    if not self.valid_trans(A, dt_1, dt_2, _cid, cid):
                        continue
                    x2, y2 = self.grid.cell2index(_cid)
                    avg_act_dist = (math.sqrt((x2-x1)**2 + (y2-y1)**2)) * self.grid.side
                    t1 = t2 = t3 = 0
                    t1 = f[idx-1][_state]
                    t3 = B[obsv][state]
                    if not (min_dist - self.grid.side * k <= avg_act_dist <= max_dist + self.grid.side * k):
                        if debug:
                            print 'idx=%d,(%d,%d)'%(idx, _cid, cid)
                            print 'dist(%d,%d) act(%d)' % (int(min_dist), int(max_dist), int(avg_act_dist))
                        continue
                    # t2 = self.transfer(_cid, cid)
                    t2 = self.transfer(A, dt_1, dt_2, _cid, cid)
                    if avg_act_dist < min_dist:
                        t2 *= math.e ** (-(avg_act_dist - min_dist)**2/(min_dist**2)/2)
                    if avg_act_dist > max_dist:
                        t2 *= math.e ** (-(avg_act_dist - max_dist)**2/(max_dist**2)/2)
                    alt_p = t1 + t2 * t3
                    if alt_p > max_p:
                        max_p = alt_p
                        p[idx][state] = _state
                    h[state] = max_p
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

    def viterbi_(self, A, B, traj, k, help_info, debug=False):
        f, p = [], defaultdict(dict)
        h = dict()
        obsv = self.get_obsv(traj[0])
        piece_help = help_info[0]
        for state, prob in B[obsv].iteritems():
            cid, big_id = self.uzip(state)
            h[state] = prob * piece_help[big_id]
        f.append(h)
        for idx in range(1, len(traj)):
            obsv = self.get_obsv(traj[idx])
            piece_help = help_info[idx]
            speed, timestp = traj[idx][1][-2:]
            h = dict()
            _obsv = self.get_obsv(traj[idx-1])
            _speed, _timestp = traj[idx-1][1][-2:]
            max_dist = max(_speed, speed) * (timestp - _timestp) + 1.0 * self.grid.side
            min_dist = min(_speed, speed) * (timestp - _timestp) - 1.0 * self.grid.side
            for state, prob in B[obsv].iteritems():
                max_p = 0
                cid, big_id = self.uzip(state)
                x1, y1 = self.grid.cell2index(cid)
                for _state in B[_obsv].iterkeys():
                    if not f[idx-1].has_key(_state):
                        continue
                    _cid, _big_id = self.uzip(_state)
                    x2, y2 = self.grid.cell2index(_cid)
                    avg_act_dist = (math.sqrt((x2-x1)**2 + (y2-y1)**2)) * self.grid.side
                    t1 = t2 = t3 = 0
                    t1 = f[idx-1][_state]
                    t3 = B[obsv][state] * piece_help[big_id]
                    if not (min_dist - self.grid.side * k <= avg_act_dist <= max_dist + self.grid.side * k):
                        if debug:
                            print 'idx=%d,(%d,%d)'%(idx, _cid, cid)
                            print 'dist(%d,%d) act(%d)' % (int(min_dist), int(max_dist), int(avg_act_dist))
                        continue
                    t2 = self.transfer(_state, state)
                    if avg_act_dist < min_dist:
                        t2 *= math.e ** (-(avg_act_dist - min_dist)**2/(min_dist**2)/2)
                    if avg_act_dist > max_dist:
                        t2 *= math.e ** (-(avg_act_dist - max_dist)**2/(max_dist**2)/2)
                    alt_p = t1 + t2 * t3
                    if alt_p > max_p:
                        max_p = alt_p
                        p[idx][state] = _state
                    h[state] = max_p
            f.append(h)
            if debug:
                print idx
        sl = []
        max_prob = max(f[-1].itervalues())
        last_s = max(f[-1].iterkeys(), key=lambda k: f[-1][k])
        for idx in range(len(traj)-1, 0, -1):
            cid, big_id = self.uzip(last_s)
            sl.append(cid)
            last_s = p[idx][last_s]
        cid, big_id = self.uzip(last_s)
        sl.append(cid)
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
            state = self.grid.utm2cell(x, y)
            traj_.append((state, obsv, speed, timestp))
        # print traj_
        state_, obsv, _speed, _timestp = traj_[0]
        x1, y1 = self.grid.cell2index(state_)
        prob = 1.0 * B[obsv][state_]
        states.append(state_)
        # print state_, prob
        for idx in range(1, len(traj_)):
            state, obsv, speed, timestp = traj_[idx]
            max_dist = max(_speed, speed) * (timestp - _timestp) + 1.0 * self.grid.side
            min_dist = min(_speed, speed) * (timestp - _timestp) - 1.0 * self.grid.side
            x2, y2 = self.grid.cell2index(state)
            avg_act_dist = (math.sqrt((x2-x1)**2 + (y2-y1)**2)) * self.grid.side
            
            transf = obsvf = 0
            transf = self.transfer(state_, state)
            obsvf = B[obsv][state]
            if not (min_dist - self.grid.side*k <= avg_act_dist <= max_dist + self.grid.side*k):
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
            try:
                self.utm_test[tr_id], self.utm_pred[tr_id] = [], []
                self.cid_test[tr_id], self.cid_pred[tr_id] = [], []
                self.gps_test[tr_id], self.gps_pred[tr_id] = [], []
                for idx, point in enumerate(traj):
                    # ground truth
                    lat, lng = matched[point[0]][4:6]
                    x, y, _, _ = utm.from_latlon(lat, lng)
                    cid = pdt[idx]
                    # prediction
                    px, py = self.grid.cell2utm(cid)

                    cx, cy = self.grid.utm2index(x, y)
                    gx, gy = utm.to_latlon(px, py, 51, 'R')
                    # for disp
                    self.utm_test[tr_id].append((x, y))
                    self.utm_pred[tr_id].append((px, py))
                    self.gps_test[tr_id].append((lat, lng))
                    self.gps_pred[tr_id].append((gx, gy))
                    self.cid_test[tr_id].append((cx, cy))
                    self.cid_pred[tr_id].append(self.grid.cell2index(cid))
                    dist = math.sqrt((px - x)**2 + (py - y)**2)
                    precision.append(dist)
                    avg += dist
                precision_[tr_id] = avg / len(traj)
            except IndexError as e:
                print 'IndexError at tr_id=', tr_id
                continue
        return precision, precision_


def predict(sag, db_data, A, B):
    cell_result = dict()
    statistic = []
    failed_ids = []
    for tr_id, traj in db_data.iteritems():
        speeds = [point[1][-2] for point in traj]
        k = math.floor(np.std(speeds))
        k0 = k
        noexcept = False
        while noexcept is False:
            noexcept = True
            try:
                sl, max_prob = sag.viterbi(A, B, traj, k)
            except:
                noexcept = False
                k += 1
        # prob = sag.backforward(A, B, traj, match_res[tr_id], k, debug=False)
        prob = 0
        statistic.append((tr_id, max_prob, prob))
        cell_result[tr_id] = sl
        if k!=k0:
            failed_ids.append(tr_id)
        # print 'TrajID=' + str(tr_id)
        # print 'Failed=', failed_ids, len(failed_ids)
    return cell_result, statistic, failed_ids