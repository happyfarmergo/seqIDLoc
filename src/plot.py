# -*- coding:utf-8 -*-
import sys
import utm
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from matplotlib.patches import Arrow
from matplotlib import lines
from collections import defaultdict

import tools

colors = [
    '#FF0000',  # 红色
    '#00FF00',  # 绿色
    '#E066FF',  # 黄色
    '#0000FF',  # 蓝色
]

jia_utm_axis = (327000, 338000, 3458000, 3465000)


def draw_points_cells(traj, towers, ca, axis):
    # draw cell towers
    radius = 70
    for ids, info in towers.iteritems():
        rncid, cellid = ids
        index, lat, lng = info[:3]
        x, y, _, _ = utm.from_latlon(lat, lng)
        if tools.outside(x, y, axis):
            continue
        circle = Circle((x, y), radius = radius, color=colors[index%len(colors)], alpha=0.5)
        ca.text(x-radius, y, '[%d]' % index, color='k', fontsize=16)
        ca.add_patch(circle)
    # draw gps points
    last_id = -1
    ranges = []
    for i in range(len(traj)):
        cur_id = traj[i][2]
        if last_id != cur_id:
            ranges.append(i)
        last_id = cur_id
    ranges.append(len(traj))

    for a, b in tools.pairwise(ranges):
        xs, ys = [], []
        for point in traj[a:b]:
            x, y, id_1 = point[:3]
            # print x, y, id_1
            xs.append(x)
            ys.append(y)
        if len(xs) == 1:
            continue
        line = lines.Line2D(xs, ys, linewidth=2, color=colors[id_1%len(colors)])
        mx, my = xs[len(xs) / 2], ys[len(xs) / 2]
        ca.text(mx, my, '%d' % id_1, color='k', fontsize=12)
        ca.add_line(line)

# [cid,...]
def draw_traj_on_cells(traj, ca, axis, color):
    xs, ys = [], []
    # for x, y in traj:
    #     xs.append(x)
    #     ys.append(y)
    # line = lines.Line2D(xs, ys, linewidth=2, color=color)
    # ca.add_line(line)
    cnt = 0
    for x, y in traj:
        circle = Circle((x, y), radius=10, color=color, alpha=0.6)
        # ca.text(x, y, str(cnt), color='k', fontsize=10)
        ca.add_patch(circle)
        cnt += 1

def draw_cells(kdtree, ca, axis, debug=False):
    cnt = 0
    for cid, bound in kdtree.iteritems():
        x0, y0, x1, y1 = bound
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        rec = Rectangle((x0, y0), width=x1 - x0, height=y1 - y0, ec='k')
        if debug:
            ca.text(mx, my, str(cid), color='k', fontsize=10)
        ca.add_patch(rec)
        # if cnt >=100:
            # break
        cnt += 1

def draw_barplot(summary, title, xlabel, ylabel):
    plt.close()
    plt.figure(figsize=(20, 10))
    plt.bar(summary.keys(), height=summary.values())
    xticks = (str(x) for x in summary.keys())
    plt.xticks(summary.keys(), xticks)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig('./display/deepin/ID1_context.png')
    # plt.show()