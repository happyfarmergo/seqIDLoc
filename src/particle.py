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

import tools

class Particle(object):
    def __init__(self, init_stat, init_weight):
        self.states = [init_stat]
        self.weight = init_weight

    @classmethod
    def create_by_prior_dist(self, obsv, D, B, N):
        counts = dict()
        for rid in D.iterkeys():
            counts[rid] = sum(D[rid].itervalues())
        total = sum(counts.itervalues())
        prob = dict()
        for rid, val in counts.iteritems():
            prob[rid] = val * 1.0 / total

        rids = prob.keys()
        probs = prob.values()
        particles = []
        init_rids = []
        for i in range(N):
            rid = np.random.choice(rids, p=probs)
            weight = B[obsv][rid] if rid in B[obsv].keys() else 0
            particle = self(rid, weight)
            particles.append(particle)
            init_rids.append(rid)
        return particles, init_rids

    def sample_new_state(self, obsv, A, B):
        last_state = self.states[-1]
        if last_state in A.keys():
            rids = A[last_state].keys()
            probs = [prob / 2.0 for prob in A[last_state].values()]
            # print rids
            # print sum(probs)
            rid = np.random.choice(rids, p=probs)
            self.weight = B[obsv][rid] * self.weight if rid in B[obsv].keys() else 0
            self.states.append(rid)
        else:
            self.weight = 0
            self.states.append(0)


def normalize(particles):
    total = sum(p.weight for p in particles)
    for p in particles:
        p.weight /= total
    # print 'max:', max(p.weight for p in particles), ' min:', min(p.weight for p in particles)

def comp_thres(particles):
    return  1.0 / sum(p.weight ** 2 for p in particles)

def resample(particles):
    new_particles = []
    N = len(particles)
    probs = [p.weight for p in particles]
    for i in range(N):
        p_idx = np.random.choice(np.arange(0, N), p=probs)
        new_particles.append(copy.deepcopy(particles[p_idx]))
    return new_particles

# Output: [rid, ...]
def particle_filter(traj, A, B, particles, Nth):
    normalize(particles)
    for i in range(1, len(traj)):
        obsv = traj[i][2:]
        j = 0
        for p in particles:
            p.sample_new_state(obsv, A, B)
            # not_modified = True
            # for r1 in A.iterkeys():
            #     if len(A[r1]) == 0:
            #         not_modified = False
            #         break
            # print 'iterate', j, ' A not_modified', not_modified
            j += 1
        normalize(particles)
        Neff = comp_thres(particles)
        # print 'Neff:', Neff
        if Neff < Nth and i < len(traj) - 1:
            particles = resample(particles)
            # print 'Resampling ... at', i
            for p in particles:
                p.weight = 1.0 / len(particles)
    particle = max(particles, key=lambda p: p.weight)
    return particle.states