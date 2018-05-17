import math

def data_length(dbs):
    lengths = []
    for i in range(len(dbs)):
        length = 0
        for j in range(len(dbs[i]) - 1):
            pre = dbs[i][j]
            now = dbs[i][j+1]
            a, b = pre[:2]
            c, d = now[:2]
            dist = math.sqrt((d - b)**2 + (c - a)**2)
            length += dist
        lengths.append(length)
    return lengths