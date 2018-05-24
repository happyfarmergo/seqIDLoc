import numpy as np
import pandas as pd
import utm
import math
import pickle
import os
import sys
import time
import datetime
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
import copy
import sacred
from sacred.observers import MongoObserver
from sacred import Experiment

import plot
import tools
import pygrid
import regresser
import split, uni_hmm
import celltower as ctcrawler
from data_cleaning import noise, load, counter

def prepare():
    global ex
    ex = Experiment('SeqIDLoc')
    ex.observers.append(MongoObserver.create(url='localhost:27017', db_name='SeqIDLoc'))
    # plt.interactive(False)

prepare()


@ex.config
def cfg():
    dataset = 'jiading'
    datatype = '2g'
    side = 30
    neighbor = False
    with_rssi = False
    context = False

@ex.automain
def main(dataset, datatype, side, neighbor, with_rssi, context, _log, _run):
    data_file = '../data/%s_%s/data_%s.csv' % (dataset, datatype, datatype)
    gongcan_file = '../data/%s_%s/gongcan_%s.csv' % (dataset, datatype, datatype)
    disp_path = '../display/%s_%s/' % (dataset, datatype)
    data_path = '../data/%s_%s/' % (dataset, datatype)
    map_file = '../data/%s_map/%s_EdgeGeometry.txt' % (dataset, dataset)
    if dataset == 'jiading':
        bounding_box = (328000, 331000, 3462000, 3465000)
    elif dataset == 'siping':
        bounding_box = (356000, 359000, 3461500, 3463000)
    print dataset, datatype, side, bounding_box

    _run.info['time'] = {}
    _run.info['side_length'] = side
    total_time = 0
    '''
    Load Raw Data Set
    '''
    db, db_gps, towers = load.load_data(data_file, gongcan_file, merge_tower=False, neighbor=neighbor, with_rssi=with_rssi, radio_angle=False, context=context)
    print 'len(db):', len(db), 'len(towers):', len(towers)

    '''
    Preprocessing
    '''
    max_dist, min_dist, min_len, max_again = noise.get_config(dataset, datatype)
    dbs = noise.clean_db(db, max_dist, min_dist, min_len, max_again, debug=False)
    dbs_gps = noise.clean_db_gps(db_gps, max_dist, min_dist, min_len, max_again, debug=False)
    print 'len(dbs):', len(dbs), 'len(dbs_gps):', len(dbs_gps)

    '''
    Map matching Result
    '''
    t0 = time.time()
    match_res, time_set = load.load_matching(data_path + 'matching_out', len(dbs), 50)
    dbs = noise.reclean(dbs, time_set, debug=False)
    dbs_gps = noise.reclean(dbs_gps, time_set)
    t1 = time.time()
    total_time += t1 - t0
    _run.info['time']['#1_Mapmatching'] = t1 - t0
    '''
    Split Dataset into Train and Test
    '''
    train, test, gps3 = split.k_g_split(dbs, k=4, gpsize=0.0)
    train_set = set()
    for tr_id, traj in train.iteritems():
        for point in traj:
            train_set.add(point[1][2:-2])
    test_set = set()
    for tr_id, traj in test.iteritems():
        for point in traj:
            test_set.add(point[1][2:-2])
    print 'Diff between train and set:', len(test_set - train_set)

    '''
    Layer 1 Train
    '''
    grid = pygrid.Grid(side, bounding_box)
    roadmap = load.load_map(map_file)
    slice_map = tools.make_map(grid, roadmap, 90, 1)

    t0 = time.time()
    sag = uni_hmm.Sag(grid, slice_map, 0)
    A, cell_db = sag.get_trans_mat(match_res, gps3.keys())
    B, D = sag.get_obsv_mat(train, match_res, version=1)
    new_test = noise.clean_testset(sag, test, match_res, B, debug=True)
    t1 = time.time()
    total_time += t1 - t0
    _run.info['time']['#2_Layer1_Offline'] = t1 - t0

    '''
    Layer 1 predict
    '''
    t0 = time.time()
    cell_train, _, _ = uni_hmm.predict(sag, train, A, B)
    print 'HMM Train Completed!'
    cell_pred, _, _ = uni_hmm.predict(sag, new_test, A, B)
    print 'HMM Test Completed!'
    t1 = time.time()
    total_time += t1 - t0
    _run.info['time']['#3_Layer1_Online'] = t1 - t0

    '''
    Layer 1 Evaluation
    '''
    uni_prec, uni_prec_dict = sag.evaluate(new_test, cell_pred, match_res)
    print np.median(uni_prec), np.median(uni_prec_dict.values()), np.mean(uni_prec), len(uni_prec_dict)

    _run.info['layer1'] = [uni_prec, uni_prec_dict, np.median(uni_prec), np.median(uni_prec_dict.values()), np.mean(uni_prec), len(uni_prec_dict)]


    X_train, Y_train = regresser.load_data(grid, train, cell_train, match_res)
    X_test, Y_test = regresser.load_data(grid, new_test, cell_pred, match_res)


    '''
    Layer 2 Train
    '''
    t0 = time.time()
    model = regresser.train(X_train, Y_train)
    print 'Regressor Train Completed!'
    t1 = time.time()
    total_time += t1 - t0
    _run.info['time']['#4_Layer2_Train'] = t1 - t0

    '''
    Layer 2 Test
    '''
    t0 = time.time()
    Y_pred = model.predict(X_test)
    print 'Regressor Predict Completed!'
    t1 = time.time()
    total_time += t1 - t0
    _run.info['time']['#4_Layer2_Predict'] = t1 - t0

    '''
    Layer 2 Evaluation
    '''
    result = regresser.evaluate(Y_test, Y_pred)
    _run.info['total_time'] = total_time
    _run.info['layer2'] = [result, np.median(result), np.mean(result)]

    print np.median(result), np.mean(result)