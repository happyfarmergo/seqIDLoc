# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import utm
import math
import os
import sys
import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict

def load_map(map_file):
    roadmap = dict()
    with open(map_file) as f:
        for line in f:
            nodes = []
            data = [float(d) for d in line.strip().split('^')[4:]]
            rid = int(line.strip().split('^')[0])
            lngs, lats = data[1::2], data[::2]
            roadmap[rid] = [(lat, lng) for lat, lng in zip(lats, lngs)]
    return roadmap

def load_gongcan(gongcan_file):
    gongcan = pd.read_csv(gongcan_file)
    # merged cell tower, origin cell tower
    m_towers, o_towers = dict(), dict()
    duplicate = defaultdict(list)
    dup_ids = []
    dup_set = set()
    for i in range(len(gongcan)):
        piece_data = gongcan.iloc[i]
        rncid, cellid, lng, lat, azimuth, downtilt = \
        piece_data['RNCID'], piece_data['CellID'], \
        piece_data['Longitude'], piece_data['Latitude'], \
        piece_data['Azimuth'], piece_data['Downtilt']
        if (lat, lng) not in duplicate.keys():
            m_towers[(rncid, cellid)] = (i, lat, lng, azimuth, downtilt)
        o_towers[(rncid, cellid)] = (i, lat, lng, azimuth, downtilt)
        duplicate[(lat, lng)].append(i)
    print 'Unique Cell Tower: ', len(m_towers)
    iterator = (ids for loc, ids in duplicate.iteritems() if len(ids) > 1)
    for ids in iterator:
        dup_ids.append(ids)
        dup_set.update(ids)
    print 'Duplicated cell towers:', dup_ids
    return m_towers, o_towers, dup_ids, dup_set

def parse_argv(feature):
    # merge_tower = feature['merge_tower']
    merge_tower = False
    neighbor = feature['neighbor']
    with_rssi = feature['with_rssi']
    # radio_angle = feature['radio_angle']
    radio_angle = False
    context = feature['context']
    return merge_tower, neighbor, with_rssi, radio_angle, context

def find_cell_id(old_id, dup_set, dup_ids):
    if old_id not in dup_set:
        return old_id
    for ids in dup_ids:
        if old_id in ids:
            return ids[0]
    
def load_data(data_file, gongcan_file, **feature):
    merge_tower, neighbor, with_rssi, radio_angle, context = parse_argv(feature)
    m_towers, o_towers, dup_ids, dup_set = load_gongcan(gongcan_file)
    towers = m_towers if merge_tower else o_towers
    db = dict()
    db_gps = dict()
    df = pd.read_csv(data_file)
    last_stp = 0
    dup_cnt = 0
    for i, piece_data in df.iterrows():
        rid_1, cid_1, rssi_1, rid_2, cid_2, rssi_2, lat, lng, speed, tr_id, timestp = \
        piece_data['RNCID_1'], piece_data['CellID_1'], piece_data['Dbm_1'], \
        piece_data['RNCID_2'], piece_data['CellID_2'], piece_data['Dbm_2'], \
        piece_data['Latitude'], piece_data['Longitude'], piece_data['Speed'], \
        int(piece_data['TrajID']), int(piece_data['MRTime']) / 1000
        if last_stp == timestp:
            dup_cnt += 1
            continue
        index_1 = o_towers[(rid_1, cid_1)][0] if (rid_1, cid_1) in o_towers.keys() else -1
        if index_1 == -1:
            print 'data error! BestCellID is wrong!', 'tr_id=%d, idx=%d, (%s, %s)' % (tr_id, i, rid_1, cid_1)
        index_2 = o_towers[(rid_2, cid_2)][0] if (rid_2, cid_2) in o_towers.keys() else -1
        x, y, _, _ = utm.from_latlon(lat, lng)
        if merge_tower:
            index_1 = find_cell_id(index_1, dup_set, dup_ids)
            index_2 = find_cell_id(index_2, dup_set, dup_ids)
        # 拼装输入数据特征
        point = (x, y, index_1, speed, timestp)
        point_gps = (lat, lng, index_1, speed, timestp)
        if neighbor:
            point = (x, y, index_1, index_2, speed, timestp)
            point_gps = (lat, lng, index_1, index_2, speed, timestp)
        if with_rssi:
            rssi_1, rssi_2 = int(rssi_1 / 10) * 10, int(rssi_2 / 10) * 10
            point = (x, y, index_1, rssi_1, speed, timestp) if not neighbor else (x, y, index_1, rssi_1, index_2, rssi_2, speed, timestp)
            point_gps = (lat, lng, index_1, rssi_1, speed, timestp) if not neighbor else (lat, lng, index_1, rssi_1, index_2, rssi_2, speed, timestp)
        if not db.has_key(tr_id):
            db[tr_id] = []
        if not db_gps.has_key(tr_id):
            db_gps[tr_id] = []
        db[tr_id].append(point)
        db_gps[tr_id].append(point_gps)
        last_stp = timestp
    print 'Totally duplicate:', dup_cnt
    if context:
        db = refeature(db)
        db_gps = refeature(db_gps)
    return db, db_gps, towers

def refeature(db):
    new_db = dict()
    for tr_id, traj in db.iteritems():
        new_traj = []
        for idx in range(len(traj)):
            fea_pre = traj[idx - 1][2:-2] if idx > 0 else traj[idx][2:-2]
            fea_cur = traj[idx][2:-2]
            fea_next = traj[idx + 1][2:-2] if idx < len(traj) - 1 else traj[idx][2:-2]
            point = []
            point.extend(traj[idx][:2])
            point.extend(list(fea_pre))
            point.extend(list(fea_cur))
            point.extend(list(fea_next))
            point.extend(traj[idx][-2:])
            new_traj.append(tuple(point))
        new_db[tr_id] = new_traj
    return new_db


def for_map_matching(dbs, out_folder):
    for tr_id, traj in dbs.iteritems():
        s_traj = ''
        for point in traj:
            lat, lng = point[:2]
            timestp = point[-1]
            s_traj += '%s,%s,%s\n' % (timestp, lat, lng)
        with open(out_folder + '/' + str(tr_id) + '.txt', 'w') as fout:
            fout.write(s_traj)

def load_matching(output_folder, max_trid, dist):
    result = dict()
    timeset = dict()
    for tr_id in range(max_trid):
        result[tr_id] = []
        timeset[tr_id] = set()
        filename = output_folder + '/' + str(tr_id) + '.txt.HMM.G_gap=5_dist=' + str(dist) + '.txt'
        with open(filename) as fin:
            for line in fin:
                data = line.strip().split(',')
                if len(data)!=8:
                    continue
                data = [float(x) for x in data]
                data[0] = int(data[0])
                data[3] = int(data[3])
                if data[4] == 0:
                    continue
                result[tr_id].append(tuple(data))
                timeset[tr_id].add(data[0])
    return result, timeset