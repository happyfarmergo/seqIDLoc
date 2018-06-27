# coding: utf-8
import numpy as np
import math as Math
import random
import time
from math import atan, cos, pi
from contextlib import contextmanager
from collections import defaultdict

timeCount = defaultdict(int)
callCount = defaultdict(int)


def rad(d):
    return d * Math.pi / 180.0


def performance(name=""):
    """Decorator function to calculate elapsed time of the function"""
    def wrapper(func):
        def _wrapper(*args, **kwargs):
            # print '' % name,
            start = time.time()
            result = func(*args, **kwargs)
            timeCount[name] += time.time()-start
            callCount[name] += 1
            # print 'running %s ... %.2fs' % (name, time.time()-start)
            return result
        return _wrapper
    return wrapper

timeit_id = 1
@contextmanager
def timeit(msg):
    """Calculate running time of a code block"""
    
    global timeit_id
    start = time.time()
    yield
    print "%d. %s elapse %.2fs" % (timeit_id, msg, time.time()-start)
    timeit_id += 1


def parallel_helper(obj, methodname, *args, **kwargs):
    """Helper to workaround Python 2 limitations of pickling instance methods"""
    return getattr(obj, methodname)(*args, **kwargs)


def applyContext(arr, f):
    """Apply function `f` to every adjacent element of `arr`"""
    return np.array([f(arr[i], arr[i+1]) for i in xrange(len(arr)-1)])


def timer(name=""):
    """Decorator function to calculate elapsed time of the function"""
    def wrapper(func):
        def _wrapper(*args, **kwargs):
            # print '' % name,
            start = time.time()
            result = func(*args, **kwargs)
            print 'running %s ... %.2fs' % (name, time.time()-start)
            return result
        return _wrapper
    return wrapper


def getCoverArea(locations, padding):
    """ Get the coverage area of given locations

    Parameters
    ----------
    locations : list
        All the points that in this area.
        Each entry is a coordinate (longitude, latitude).
    padding : int
        Distance between the area edge and the minimum rectanlge
        that cover all the locations.

    Returns
    -------
    area : tuple, shape=((minLon, minLat), (maxLon, maxLat))
        The left down corner and right up corner of the area.
    """
    lonStep_1m = 0.0000105
    latStep_1m = 0.0000090201
    longitudes = locations[:, 0]
    latitudes = locations[:, 1]
    minLon = np.min(longitudes) - padding * lonStep_1m
    maxLon = np.max(longitudes) + padding * lonStep_1m
    minLat = np.min(latitudes) - padding * latStep_1m
    maxLat = np.max(latitudes) + padding * latStep_1m
    return ((minLon, minLat), (maxLon, maxLat))


def azimuth(pt_a, pt_b):
    """ Compute azimuth from `pt_a` to `pt_b`

    Parameters
    ----------
    pt_a : tuple or list, shape = (Longitude, Latitude)
        Coordinate
    pt_b : tuple or list, shape = (Longitude, Latitude)
        Coordinate

    Returns
    -------
    angle : float
        Azimuth from `pt_a` to `pt_b`
    """
    rc = 6378137
    rj = 6356725
    lon_a, lat_a = pt_a
    lon_b, lat_b = pt_b
    rlon_a, rlat_a = rad(lon_a), rad(lat_a)
    rlon_b, rlat_b = rad(lon_b), rad(lat_b)
    ec = rj+(rc-rj)*(90.-lat_a)/90.
    ed = ec*cos(rlat_a)

    dx = (rlon_b - rlon_a) * ec
    dy = (rlat_b - rlat_a) * ed
    if dy == 0:
        angle = 90.
    else:
        angle = atan(abs(dx / dy)) * 180.0 / pi
    dlon = lon_b - lon_a
    dlat = lat_b - lat_a
    if dlon > 0 and dlat <= 0:
        angle = (90. - angle) + 90
    elif dlon <= 0 and dlat < 0:
        angle = angle + 180
    elif dlon < 0 and dlat >= 0:
        angle = (90. - angle) + 270
    return angle


def distance(point1, point2):
    """ Compute earth distance between to coordinate (longitude, latitude)

    Parameters
    ----------
    point1 : tuple or list. shape=(Longitude, Latitude)
        One coordinate.
    point2 : tuple or list. shape=(Longitude, Latitude)
        One coordinate.

    Returns
    -------
    dist : int
        Earth distance between `point1` and `point2` (meter).
    """
    lat1 = float(point1[1])
    lng1 = float(point1[0])
    lat2 = float(point2[1])
    lng2 = float(point2[0])
    radLat1 = rad(lat1)
    radLat2 = rad(lat2)
    a = radLat1 - radLat2
    b = rad(lng1) - rad(lng2)
    dist = 2 * Math.asin(Math.sqrt(Math.pow(Math.sin(a/2), 2) +
                                   Math.cos(radLat1) *
                                   Math.cos(radLat2) *
                                   Math.pow(Math.sin(b/2), 2)))
    dist = dist * 6378.137
    dist = round(dist * 10000) / 10
    return dist


def constructTrajectories(times,
                          locations=None,
                          maxTrajLength=-1,
                          minDist=-1,
                          timeGap=10):
    """
    Construct trajectories from timestamp list of data.
    The distance of every two adjacent points should less than a threshold and
    the time difference should also less than a threshold.

    Parameters
    ----------
    times : array_like
        the timestamp list of trajectory data
        each timestamp's format is "yyyy-MM-dd HH:mm:SS.fff"
    locations : array_like, default=None
        the list of locations
        each element is longitude/latitude
    maxTrajLength : int, default=-1
        the maximum length of a extracted trajectory
        if maxTrajLength=-1, then do not limit
    timeGap : int, default=10
        the maximum time (second) between two adjacent points
        in one trajectory

    Return
    ------
    trajs : list
        each element of `trajs` is a trajectory, which is a index list
        (the index is the index of `times` or `locations`)
        for example,
            [[1,2,3,4,5,6],
             [7,8,9,10,11,12],
             ...
            ]
    """
    if type(times[0]) == str:
        times = [toUnixTime(t, '%Y-%m-%d %H:%M:%S.%f') for t in times]
    orderedId = np.argsort(times)

    maxTrajLength = maxTrajLength if maxTrajLength != -1 else len(times)

    trajs, oneTraj = [], []
    trajs_time, oneTraj_time = [], []
    for i in orderedId:

        if len(oneTraj) == 0:
            oneTraj.append(i)
            # oneTraj_time.append(times[i])
        # the return trajectory's length should no larger than `maxTrajLength`
        elif len(oneTraj) >= maxTrajLength:
            trajs.append(oneTraj)
            trajs_time.append(oneTraj_time)
            oneTraj = [i]
            # oneTraj_time = [times[i]]
        else:
            lastId = oneTraj[len(oneTraj)-1]
            if locations is None:
                locationCondition = True
            else:
                lastLocation = locations[lastId]
                currentLocation = locations[i]
                locationCondition = (
                    minDist < distance(lastLocation, currentLocation) <= 100)

            lastTime = times[lastId]
            currentTime = times[i]

            # check the distance and time gap between ajacent points
            if  locationCondition and \
                    currentTime - lastTime <= timeGap:
                oneTraj.append(i)
                # oneTrajtimes, locations_time.append(times[i])
            else:
                trajs.append(oneTraj)
                # trajs_time.append(oneTraj_time)
                oneTraj = [i]
                oneTraj_time = [times[i]]

    # put remaining points into `trajs`
    if len(oneTraj) > 0:
        trajs.append(oneTraj)
        # trajs_time.append(oneTraj_time)
    # return trajs, trajs_time
    return trajs


def toUnixTime(t, format='%Y-%m-%d %H:%M:%S.%f'):
    """ Convert time string to timestamp

    Parameters
    ----------
    t : str
        Time string. For example, 2016-12-01 13:01:05
    format : str. default='%Y-%m-%d %H:%M:%S'
        The format of time string.
    """
    try:
        return int(time.mktime(time.strptime(t, format)))
    except Exception as e:
        return -1 

'''
def gaussianDistribution(x, mean, std):
    """
    Compute a value of Gaussian Distribution

    Parameters
    ----------
    x : float
        variable of Gaussian Distribution
    mean : float
        mean value of Gaussian Distribution
    std : float
        standard deviation of Gaussian Distribution

    Return
    ------
    proba : float
        the probability in Gaussian Distribution with variable x
    """
    return (1 / ((2*np.pi) ** 0.5)*std) * \
        np.exp(- (x - mean) ** 2 / (2 * std**2))
'''

def normalize(arr):
    """
    Normalized an array to satisfied \sum_i arr_i = 1

    Parameters
    ----------
    arr : array_like
        array to be normalized.
        The elements' data type should be numerical (int, float, etc.)

    Returns
    -------
    normalized_arr : np.array
        array that is normalized
    """
    arr = np.array(arr, dtype=np.float32)
    return arr / np.sum(arr)


# def sampleByDistribution(distribution, n=1):
#     """
#     Sample x from a discrete distribution

#     Parameters
#     ----------
#     distribution : array like
#             a discrete distribution

#     Returns
#     -------
#     int
#         a value x (0 <= x < len(distribution)) sampled from the distribution
#     """
#     distribution = normalize(
#         distribution)  # make sure this satisfied the distribution constraint
#     # accumDistribution = np.hstack(([0], np.array(distribution).cumsum()))

#     samples = []
#     mw = max(distribution)
#     N = len(distribution)
#     beta = 0.0
#     index = int(random.random() * N)
#     for _ in range(n):
#         # divided by 1.0001 in order to make sure randomNumber < 1
#         # randomNumber = random.random() / 1.0001
#         # l = 0
#         # r = len(accumDistribution)
#         # while l+1 < r:
#         #     m = int((l + r) / 2)
#         #     if accumDistribution[m] <= randomNumber:
#         #         l = m
#         #     else:
#         #         r = m
#         # samples.append(l)
#         beta += random.random() * 2.0 * mw
#         while beta > distribution[index]:
#             beta -= distribution[index]
#             index = (index + 1) % N
#         samples.append(index)
#     if n == 1:
#         return samples[0]
#     else:
#         return samples


def binary_search(arr, val, key=lambda x: x):
    """ Binary search

    Parameters
    ----------
    arr : array_like
        The array to search, which should be sorted.
    val : int or float
        Target value.
    key : function, default=lambda x: x
        The transform of each val in `arr`

    Returns
    -------
    idx : int
        The first index that satisfied `arr[index] >= val`
    """
    l, r = 0, len(arr)
    while l < r:
        m = int((l+r) / 2)
        if key(arr[m]) < val:
            l = m+1
        else:
            r = m
    return l
