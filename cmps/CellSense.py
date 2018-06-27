# coding: utf-8
import numpy as np
import math as Math


rc = 6378137
rj = 6356725

def rad(d):
    return d * Math.pi / 180.0

def distance(true_pt, pred_pt):
    lat1 = float(true_pt[1])
    lng1 = float(true_pt[0])
    lat2 = float(pred_pt[1])
    lng2 = float(pred_pt[0])
    radLat1 = rad(lat1)
    radLat2 = rad(lat2)
    a = radLat1 - radLat2
    b = rad(lng1) - rad(lng2)
    s = 2 * Math.asin(Math.sqrt(Math.pow(Math.sin(a/2),2) +
    Math.cos(radLat1)*Math.cos(radLat2)*Math.pow(Math.sin(b/2),2)))
    s = s * 6378.137
    s = round(s * 10000) / 10
    return s

class FingerPrint:
    def __init__(self, id_list, strength_list, lnglat=None):
        assert(len(id_list) == len(strength_list))
        self.id_list = id_list
        self.strength_list = strength_list
        self.lnglat = lnglat

'''
class Histogram:
    def __init__(self, data, n_bins):
        self.n_bins = n_bins
        bins = range(np.min(data), np.max(data),2)+[np.max(data)+1]
        x, y = np.histogram(data, bins=bins)
        self.n_bins = len(bins)-1
        self.hist = 1.0 * x / len(x)
        self.bin_edges = y

    def get_proba(self, v):
        # OPTIMIZE LATER
        if v < self.bin_edges[0] or v >= self.bin_edges[-1]:
            return 0.
        else:
            for i in range(self.n_bins-1):
                if self.bin_edges[i] <= v < self.bin_edges[i+1]:
                    return self.hist[i]
        return 0.
'''

class Histogram:
    def __init__(self, data, default_std):
        self.mean = np.mean(data)
        self.std = np.std(data)
        if self.std == 0:
            self.std = default_std

    def get_proba(self, v):
        return np.exp(-(v-self.mean)**2/(self.std**2))/(np.sqrt(2*np.pi)*self.std)


class Grid:
    def __init__(self, ld, ru):
        self.ld = ld
        self.ru = ru
        self.stats = {}
        self.points = []

        self.lnglat = None
        self.hist = {}

    def in_grid(self, x, y):
        return self.ld[0] <= x < self.ru[0] and self.ld[1] <= y < self.ru[1]

    def compute_histogram(self, default_std):
        self.hist = {}
        for cid in self.stats.keys():
            self.hist[cid] = Histogram(self.stats[cid], default_std)
        self.lnglat = tuple(np.mean(self.points, axis=0))

    def add_fingerpoint(self, fp):
        for cid, strength in zip(fp.id_list, fp.strength_list):
            self.points.append(fp.lnglat)
            if not self.stats.has_key(cid):
                self.stats[cid] = []
            self.stats[cid].append(strength)

    def add_fingerpoints(self, fps):
        for fp in fps:
            self.add_fingerpoint(fp)

    def get_proba(self, fp):
        if len(self.hist) > 0:
            proba = 1.
            for cid, strength in zip(fp.id_list, fp.strength_list):
                if cid not in self.hist.keys():
                    proba *= 0.0001
                    continue
                h = self.hist[cid]
                proba *= h.get_proba(strength)
            return proba
        else:
            print 'Please call compute_histogram first'

    def get_probas(self, fps, inital,is_c=True):
        if is_c:
            return [self.get_proba(fp) for fp in fps]
        else:
            return [self.get_proba(fp)*inital for fp in fps]

class CellSense:
    def __init__(self, area, grid_length, default_std, datatype):
        self.ld, self.ru = area
        self.grid_length = grid_length
        self.default_std = default_std
        self.vsize = distance(self.ld, (self.ld[0], self.ru[1])) / grid_length
        self.hsize = distance(self.ld, (self.ru[0], self.ld[1])) / grid_length
        self.glng_offset = (self.ru[0]-self.ld[0]) / grid_length
        self.glat_offset = (self.ru[1]-self.ld[1]) / grid_length
        self.is4g = datatype == '4g'

        self.grid = {}
        self.inital={}

    def get_grid(self, lnglat):
        x, y = lnglat
        if self.ld[0] <= x < self.ru[0] and self.ld[1] <= y < self.ru[1]:
            i = int(distance(self.ld, (x, self.ld[1])) / self.grid_length)
            j = int(distance(self.ld, (self.ld[0], y)) / self.grid_length)

            gx = self.ld[0] + i*self.glng_offset
            gy = self.ld[1] + j*self.glat_offset
            return i + j * self.hsize, (gx, gy), (gx+self.glng_offset, gy+self.glat_offset)
        else: # not in this grid
            return -1, -1, -1

    def transform(self, df):
        data = []
        for idx, row in df.iterrows():
            id_list = []
            st_list = []
            for i in range(1, 8):
                lac = row['RNCID_%d'%(i)]
                ci = row['CellID_%d'%(i)]
                rssi = row['Dbm_%d'%(i)]
                #rscp = row['RSCP_%d'%(i)]
                try:
                    if self.is4g:
                        if Math.isnan(lac) or Math.isnan(ci) or Math.isnan(rssi):
                            continue
                        if int(lac) == -1 or int(ci) == -1 or int(rssi) == -1:
                            continue
                        lac_, ci_ = lac, ci
                        if int(lac) == 2147483647 or int(ci) == 2147483647:
                            lac_, ci_ = i, i
                        id_list.append('%d:%d'%(int(lac_), int(ci_)))
                        st_list.append(rssi)
                    else:
                        if Math.isnan(lac) or Math.isnan(ci) or Math.isnan(rssi):
                            continue
                        if int(lac) == -999 or int(ci) == -999 or int(rssi) == -999:
                            continue
                        id_list.append('%d:%d'%(int(lac),int(ci)))
                        st_list.append(rssi)
                except Exception, _data:
                    print _data, lac, ci, idx
                    exit()
            data.append(FingerPrint(id_list, st_list, (row['Longitude'], row['Latitude'])))
        return data

    def fit(self, data, transform_func=None):
        fps = transform_func(data) if transform_func is not None else self.transform(data)
        for fp in fps:
            gid, gld, gru = self.get_grid(fp.lnglat)
            if gid != -1:
                if self.grid.has_key(gid) is not True:
                    self.grid[gid] = Grid(gld, gru)
                self.grid[gid].add_fingerpoint(fp)
        # compute histograms
        #inital=dict()
        for gid in self.grid.keys():
            self.inital[gid]=0
        i=0
        for fp in fps:
            gid, gld, gru=self.get_grid(fp.lnglat)
            if gid !=-1:
                if self.inital.has_key(gid) is True:
                    i+=1
                    self.inital[gid]+=1
        for gid in self.inital.keys():
            self.inital[gid]=float(self.inital[gid])/float(i)
        for gid in self.grid.keys():
            self.grid[gid].compute_histogram(self.default_std)

    def predict(self, data, transform_func=None, is_c=True):
        fps = transform_func(data) if transform_func is not None else self.transform(data)
        scores = []
        for gid in self.grid.keys():
            g = self.grid[gid]
            scores.append(g.get_probas(fps,self.inital[gid],is_c))
        #scores = [self.grid[gid].get_probas(fps) for gid in self.grid.keys()]

        pred_gids = np.argmax(scores, axis=0)
        pred = [self.grid[self.grid.keys()[idx]].lnglat for idx in pred_gids]
        return scores, pred