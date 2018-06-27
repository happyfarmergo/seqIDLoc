from __future__ import print_function
import os
import pp
import math
import itertools
from collections import namedtuple

def gen_cmd(dataset, datatype, side, neighbor, with_rssi, context):
    cmd = 'python sacred_hmm.py with dataset={} datatype={} side={} neighbor={} with_rssi={} context={}' \
        .format(dataset, datatype, side, neighbor, with_rssi, context)
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
                        ['dataset', 'datatype', 'side', 'neighbor', 'with_rssi', 'context'])

    dataset = ['jiading', 'siping']
    datatype = ['2g', '4g']
    side = [15, 20, 30, 50, 100]
    neighbor = [True]
    with_rssi = [0.0]
    context = [False]

    index = 0
    for param in itertools.product(dataset, datatype, side, neighbor, with_rssi, context):
        config = Config(*param)
        cmd = gen_cmd(*config)
        if config['dataset'] == 'jiading' and config['datatype'] == '2g':
            continue
        index += 1
        jobs.append([index, cmd, job_server.submit(execute, (cmd,), )])

    for index, cmd, job in jobs:
        print("{}/{}: {} Return: {}".format(index, len(jobs), cmd, job()))

    job_server.print_stats()
