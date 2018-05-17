import utm
import math
import tools

def bounding_box(traj, a, b):
    coors = []
    if float(traj[a][0]) < 1000:
        for point in traj[a:b]:
            x, y, _, _ = utm.from_latlon(float(point[0]), float(point[1]))
            coors.append((x, y))
    else:
        coors = [(point[0], point[1]) for point in traj[a:b]]
    lats = [point[0] for point in coors]
    lngs = [point[1] for point in coors]
    x0, x1 = min(lats), max(lats)
    y0, y1 = min(lngs), max(lngs)
    return math.sqrt((x1-x0)**2 + (y1-y0)**2)

def clean_db_gps(db, max_dist=200, min_dist=100, min_len=10, max_again=10, debug=False):
    dbs = dict()
    idx = 0
    for db_i in range(len(db)):
        traj = db[db_i]
        ranges = [0]
        for i in range(len(traj)-1):
            lat1, lng1 = traj[i][0], traj[i][1]
            lat2, lng2 = traj[i + 1][0], traj[i + 1][1]
            x1, y1, _, _ = utm.from_latlon(float(lat1), float(lng1))
            x2, y2, _, _ = utm.from_latlon(float(lat2), float(lng2))
            dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if dist > max_dist:
                ranges.append(i + 1)
        ranges.append(len(traj))
        out = 'Traj ID=%d\n' % db_i
        for a, b in tools.pairwise(ranges):
            bounding = bounding_box(traj, a, b)
            if b - a < min_len or bounding < min_dist:
                out += 'discard[%d:%d]\t' % (a, b-1)
                continue
            c, d, delta = trim_traj(traj, a, b)
            dbs[idx] = traj[c:d] if delta > max_again else traj[a:b]
            out += 'keep[%d:%d]as id=%d\t' % (c, d-1, idx)
            idx += 1
        if debug:
            print out
    return dbs

def clean_db(db, max_dist=200, min_dist=100, min_len=10, max_again=10, debug=False):
    dbs = dict()
    idx = 0
    for db_i in range(len(db)):
        traj = db[db_i]
        ranges = [0]
        out = 'Traj ID=%d\n' % db_i
        for i in range(len(traj)-1):
            x1, y1 = traj[i][0], traj[i][1]
            x2, y2 = traj[i + 1][0], traj[i + 1][1]
            dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if dist > max_dist:
                out += '%d->%d:%sm\t' % (i, i+1, int(dist))
                ranges.append(i + 1)
        ranges.append(len(traj))
        out += '\n'
        for a, b in tools.pairwise(ranges):
            bounding = bounding_box(traj, a, b)
            if b - a < min_len or bounding < min_dist:
                out += 'discard[%d:%d]=%dm\t' % (a, b-1, int(bounding))
                continue
            c, d, delta = trim_traj(traj, a, b)
            dbs[idx] = traj[c:d] if delta > max_again else traj[a:b]
            if delta > max_again:
                out += 'trimed[%d,%d] ' % (c - a, b - d)
            out += 'keep[%d:%d]as id=%d\t' % (c, d-1, idx)
            idx += 1
        if debug:
            print out
    return dbs

def trim_traj(traj, a, b):
    delta = 0
    b = b - 1
    while traj[a][:2] == traj[a+1][:2]:
        a += 1
        delta += 1
    while traj[b][:2] == traj[b-1][:2]:
        b -= 1
        delta += 1
    assert a < b
    return a, b + 1, delta

def reclean(dbs, time_set, debug=False):
    new_dbs = dict()
    for tr_id, traj in dbs.iteritems():
        tset = time_set[tr_id]
        new_dbs[tr_id] = []
        has_not_match = False
        for point in traj:
            if point[-1] not in tset:
                has_not_match = True
                continue
            new_dbs[tr_id].append(point)
        if debug and has_not_match:
            print 'traj id=', tr_id, len(traj), len(new_dbs[tr_id])
    return new_dbs

def clean_testset(sag, test, match_res, B, debug=False):
    new_test = dict()
    for tr_id in test.iterkeys():
        traj = test[tr_id]
        matched = match_res[tr_id]
        new_points = [] 
        for idx, point in enumerate(traj):
            piece_match = matched[point[0]]
            assert piece_match[0] == point[1][-1]
            rid, lat, lng = piece_match[3], piece_match[4], piece_match[5]
            x, y, _, _ = utm.from_latlon(lat, lng)
            cid = sag.utm2cell(x, y)
            obsv = sag.get_obsv(point)
            state = (cid, rid)
            if B.has_key(obsv) and B[obsv].has_key(state):
                new_points.append(point)
        new_test[tr_id] = new_points
        if len(traj)!=len(new_points) and debug:
            print 'TrajID=%d Before=%d End=%d' % (tr_id, len(traj), len(new_test[tr_id]))
    return new_test