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
import tools
from collections import defaultdict

class Grid:
    def __init__(self, side, bounding_box):
        self.side = side
        self.x0, self.x1, self.y0, self.y1 = bounding_box
        self.width = int(math.ceil((self.x1 - self.x0) / self.side))
        self.height = int(math.ceil((self.y1 - self.y0) / self.side))

    def utm2index(self, x, y):
        ix = int((x - self.x0) / self.side)
        iy = int((y - self.y0) / self.side)
        return ix, iy

    def utm_outside(self, x, y):
        return x < self.x0 or x > self.x1 or y < self.y0 or y > self.y1

    def utm2cell(self, x, y):
        if self.utm_outside(x, y):
            raise ValueError('utm outside')
        ix, iy = self.utm2index(x, y)
        return self.index2cell(ix, iy)

    def index2cell(self, ix, iy):
        return ix + iy * self.width

    def cell2index(self, cid):
        y = int(cid / self.width)
        x = cid % self.width
        return x, y

    def cell2utm(self, cid):
        x0, y0, x1, y1 = self.cell2box(cid)
        return (x0 + x1) / 2.0, (y0 + y1) / 2.0

    def cell2box(self, cid):
        x, y = self.cell2index(cid)
        x0, y0 = x * self.side + self.x0, y * self.side + self.y0
        return x0, y0, x0 + self.side, y0 + self.side
        
    def cell_vec(self, c1, c2):
        x1, y1 = self.cell2index(c1)
        x2, y2 = self.cell2index(c2)
        return x2 - x1, y2 - y1