from __future__ import print_function
import os
import pp
import math
import itertools
from collections import namedtuple

def gen_cmd(dataset, datatype, side, def_std, test_size):
    cmd = 'python sacred_sense.py with dataset={} datatype={} side={} def_std={} test_size={}' \
        .format(dataset, datatype, side, def_std, test_size)
    return cmd


def execute(cmd):
    print(cmd)
    res = os.popen(cmd).read()
    print(res)
    return True


if __name__ == '__main__':
    ppserver = ()
    job_server = pp.Server(ppservers=ppserver)
    print("Starting pp with " + str(job_server.get_ncpus()) + " workers")

    Config = namedtuple('Config',
                        ['dataset', 'datatype', 'side', 'def_std', 'test_size'])

    dataset = ['jiading', 'siping']
    datatype = ['2g', '4g']
    side = [10, 15, 20, 25, 30, 40]
    def_std = [5]
    test_size = [0.33, 0.25]

    for param in itertools.product(dataset, datatype, side, def_std, test_size):
        config = Config(*param)
        cmd = gen_cmd(*config)
        index += 1
        jobs.append([index, cmd, job_server.submit(execute, (cmd,), )])

    for index, cmd, job in jobs:
        print("{}/{}: {} Return: {}".format(index, len(jobs), cmd, job()))

    job_server.print_stats()
