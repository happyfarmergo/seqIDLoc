import numpy as np
import pandas as pd
import utm
import math
import pickle
import os
import sys
import time
import datetime

import sacred
from sacred.observers import MongoObserver
from sacred import Experiment

import CellSense
from sklearn.model_selection import train_test_split

def prepare():
    global ex
    ex = Experiment('CellSense')
    ex.observers.append(MongoObserver.create(url='localhost:27017', db_name='CellSense'))
    # plt.interactive(False)

prepare()


@ex.config
def cfg():
    dataset = 'jiading'
    datatype = '2g'
    side = 20
    def_std = 5
    test_size = 0.33

@ex.automain
def main(dataset, datatype, side, def_std, test_size, _log, _run):
    # data config
    data_file = '../data/%s_%s/data_%s.csv' % (dataset, datatype, datatype)
    db = pd.read_csv(data_file)
    x0, x1, y0, y1 = min(db['Latitude']), max(db['Latitude']), min(db['Longitude']), max(db['Longitude'])
    bounding_box = ((y0, x0), (y1, x1))
    print dataset, datatype, side, bounding_box

    # remove duplicate
    if dataset == 'siping':
        last_loc = (0,0)
        row_ids = []
        for ix, row in db.iterrows():
            cur_loc = tuple(row[['Longitude', 'Latitude']].values)
            if cur_loc == last_loc:
                dup_cnt +=1
                if dup_cnt > 2:
                    continue
            else:
                dup_cnt = 0
            row_ids.append(ix)
            last_loc = cur_loc
        db = db.iloc[row_ids]

    # Split data
    X_train, X_test, Y_train, Y_test = train_test_split(db, db[['Longitude', 'Latitude']], test_size=test_size)

    # model train
    cs = CellSense.CellSense(bounding_box, side, def_std, datatype)
    t0 = time.time()
    print 'Training...'
    cs.fit(X_train)
    # model predict
    print 'Predicting...'
    score, Y_pred = cs.predict(X_test, is_c=True)
    print 'Completed.'
    t1 = time.time()
    _run.info['total_time'] = t1 - t0
    # evaluate
    error = [CellSense.distance(pt1, pt2) for pt1, pt2 in zip(Y_test.values.tolist(), Y_pred)]

    _run.info['result'] = [error, np.median(error), np.mean(error)]
    print np.median(error), np.mean(error)