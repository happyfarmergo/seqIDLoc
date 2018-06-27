import numpy as np
import pandas as pd
import utm
import math
import pickle
import os
import sys
import time

def getEngPara(filePath):
    engPara = pd.read_csv(filePath)
    engParaLonLat = engPara.drop(['Azimuth', 'Downtilt'], axis=1)
    engParaLonLat.rename(columns={'Longitude': 'lon', 'Latitude': 'lat', 'RNCID': 'LAC', 'CellID': 'CI'}, inplace=True)
    return engParaLonLat

def getBasicFeature(df, engPara, nNeighborCells=6, splitIdFeature=False):
    col_name = ['%s_%d' % (name, i) for i in range(
        1, nNeighborCells + 1) for name in ['RNCID', 'CellID', 'Dbm']]
    feature = df[col_name]

    for i in range(1, nNeighborCells + 1):
        feature = feature.merge(engPara,
                                left_on=['RNCID_%d' % i, 'CellID_%d' % i],
                                right_on=['LAC', 'CI'],
                                how='left',
                                suffixes=('', '_%d' % i))
        feature = feature.drop(['LAC', 'CI'], axis=1)
    feature.rename(columns={'lon': 'lon_1', 'lat': 'lat_1'}, inplace=True)
    feature = feature.fillna(-999)
    if not splitIdFeature:
        return feature
    else:
        idFeature = []
        for i in range(1, nNeighborCells + 1):
            idFeature.append(['%.0f,%.0f' % (x[0], x[1]) for x in
                              feature[['RNCID_%d' % i, 'CellID_%d' % i]].values])
            feature = feature.drop(['RNCID_%d' % i, 'CellID_%d' % i], axis=1)
        return feature, np.array(idFeature).T