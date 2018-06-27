import numpy as np
import pandas as pd
import utm
import math
import pickle
import os
import sys
import time
import datetime
from ContextRandomForestRegression import *
from feature_engineer import *

import sacred
from sacred.observers import MongoObserver
from sacred import Experiment

def prepare():
    global ex
    ex = Experiment('ContextRF')
    ex.observers.append(MongoObserver.create(url='localhost:27017', db_name='ContextRF'))
    # plt.interactive(False)

prepare()

def split(df, k=4):
    max_trid = max(df['TrajID'])
    test_indices, train_indices = [], []
    steps = 0
    for tr_id in range(max_trid + 1):
        length = len(df[df['TrajID']==tr_id])
        for idx in range(0, length, k):
            rand = random.randint(0, k-1) + idx
            for i in range(idx, min(idx + k, length)):
                if i!=rand:
                    train_indices.append(i + steps)
                else:
                    test_indices.append(i + steps)
        steps += length
    df_train, df_test = df.iloc[train_indices], df.iloc[test_indices]
    return df_train, df_test

def make_index(df):
    max_trid = max(df['TrajID'])
    idx = []
    cnt = 0
    for tr_id in range(max_trid + 1):
        length = len(df[df['TrajID']==tr_id])
        idx.append([])
        idx[-1] = range(cnt, cnt + length)
        cnt += length
    return idx

@ex.config
def cfg():
    dataset = 'jiading'
    datatype = '2g'
    k = 4

@ex.automain
def main(dataset, datatype, k, _log, _run):
    # data config
    data_file = '../data/%s_%s/data_%s.csv' % (dataset, datatype, datatype)
    gongcan_file = '../data/%s_%s/gongcan_%s.csv' % (dataset, datatype, datatype)
    df = pd.read_csv(data_file)
    print dataset, datatype, k
    # remove duplicate
    # if dataset == 'siping':
    #     last_loc = (0,0)
    #     row_ids = []
    #     for ix, row in df.iterrows():
    #         cur_loc = tuple(row[['Longitude', 'Latitude']].values)
    #         if cur_loc == last_loc:
    #             dup_cnt +=1
    #             if dup_cnt > 10:
    #                 continue
    #         else:
    #             dup_cnt = 0
    #         row_ids.append(ix)
    #         last_loc = cur_loc
    #     df = df.iloc[row_ids]

    engPara = getEngPara(gongcan_file)
    tr_df, te_df = split(df, k=4)

    tr_idx = make_index(tr_df)
    te_idx = make_index(te_df)
    X_train, Y_train = getBasicFeature(tr_df, engPara), tr_df[['Latitude', 'Longitude']]
    X_test, Y_test = getBasicFeature(te_df, engPara), te_df[['Latitude', 'Longitude']]

    t0 = time.time()
    crf = ContextRandomForestRegressor(nEstimators=200)
    print 'Train Model ...'
    crf.fit(X_train, Y_train, tr_df['MRTime']/1000, tr_idx)
    print 'Test Model ...'
    te_pred = crf.predict(X_test, te_df[['MRTime']]/1000, te_idx)
    print 'Evaluate ...'
    t1 = time.time()
    _run.info['total_time'] = t1 - t0
    errors = crf.evaluate(Y_test, te_pred)
    _run.info['result'] = [errors, np.median(errors), np.mean(errors)]

    print np.median(errors), np.mean(errors)