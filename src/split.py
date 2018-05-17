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


def k_split(trajs, k=3):
    train, test = dict(), dict()
    for tr_id, traj in trajs.iteritems():
        l1, l2 = [], []
        for idx in range(0, len(traj), k):
            rand = random.randint(0, k-1) + idx
            for i in range(idx, min(idx + k, len(traj))):
                if i != rand:
                    l1.append((i, traj[i]))
                else:
                    l2.append((i, traj[rand]))
        train[tr_id] = l1
        test[tr_id] = l2
    return train, test

def simple_split(trajs, k=3):
    train, test = dict(), dict()
    ids = trajs.keys()
    train_ids = set(random.sample(ids, len(ids)/k*(k-1)))
    for tr_id, traj in trajs.iteritems():
        if tr_id in train_ids:
            train[tr_id] = traj
        else:
            test[tr_id] = traj
    return train, test

def k_g_split(trajs, k=4, gpsize=0.4):
    gps3 = dict()
    other = dict()
    ids = trajs.keys()
    gpsids = set(random.sample(ids, int(len(ids) * gpsize)))
    for tr_id, traj in trajs.iteritems():
        if tr_id in gpsids:
            gps3[tr_id] = trajs[tr_id]
        else:
            other[tr_id] = trajs[tr_id]
    train, test = k_split(other, k)
    return train, test, gps3


def hand_split(trajs, k=3):
    indicies = [57, 61]
    assert k <= len(indicies)
    test_list = indicies[:k]
    train, test = dict(), dict()
    for tr_id, traj in trajs.iteritems():
        if tr_id in test_list:
            test[tr_id] = [(idx, point) for idx, point in enumerate(traj)]
        else:
            train[tr_id] = [(idx, point) for idx, point in enumerate(traj)]
    return train, test
