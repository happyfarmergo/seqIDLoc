import utm
import math
from sklearn.ensemble import RandomForestRegressor

def get_piece_data(grid, idx, traj, matched, pdt):
    old_idx, point = traj[idx]
    # ground truth
    lat, lng = matched[old_idx][4:6]
    x, y, _, _ = utm.from_latlon(lat, lng)
    # predict result
    p_x, p_y = grid.cell2utm(pdt[idx])
    return x, y, p_x, p_y, point[-2], point[-1]

def load_data(grid, db_data, predict, match_res):
    X, Y = [], []
    for tr_id, pdt in predict.iteritems():
        traj = db_data[tr_id]
        matched = match_res[tr_id]
        for idx in range(len(traj)):
            pre_idx = idx - 1 if idx > 0 else idx
            next_idx = idx + 1 if idx < len(traj)-1 else idx
            pre_feature = traj[pre_idx][1][2:-2]
            next_feature = traj[next_idx][1][2:-2]
            cur_feature = traj[idx][1][2:-2]
            cur_x, cur_y, cur_p_x, cur_p_y, cur_v, cur_t = get_piece_data(grid, idx, traj, matched, pdt)
            last_x, last_y, last_p_x, last_p_y, last_v, last_t = get_piece_data(grid, pre_idx, traj, matched, pdt)
            next_x, next_y, next_p_x, next_p_y, next_v, next_t = get_piece_data(grid, next_idx, traj, matched, pdt)
            last_dist = math.sqrt((cur_p_x - last_p_x)**2+(cur_p_y - last_p_y)**2)
            next_dist = math.sqrt((cur_p_x - next_p_x)**2+(cur_p_y - next_p_y)**2)
            last_speed_delta = cur_v - last_v
            next_speed_delta = next_v - cur_v
            last_time_gap = cur_t - last_t
            next_time_gap = next_t - cur_t
            last_direction = (cur_p_y - last_p_y) / (cur_p_x - last_p_x) if (cur_p_x - last_p_x)!=0 else 999999
            next_direction = (next_p_y - cur_p_y) / (next_p_x - cur_p_x) if (next_p_x - cur_p_x)!=0 else 999999

            # re-organize feature
            feature = list(cur_feature + pre_feature + next_feature)
            other = [cur_p_x, cur_p_y, cur_v, \
                      last_p_x, last_p_y, \
                      next_p_x, next_p_y]
            feature.extend(other)
            X.append(tuple(feature))
            Y.append((cur_x, cur_y))
    return X, Y

def train(X, Y):
    model = RandomForestRegressor(n_estimators=200)
    model.fit(X, Y)
    return model

def evaluate(Y_test, Y_pred):
    result = []
    for idx, loc in enumerate(Y_pred):
        pred_x, pred_y = loc
        real_x, real_y = Y_test[idx]
        result.append(math.sqrt((pred_x - real_x)**2 + (pred_y - real_y)**2))
    return result