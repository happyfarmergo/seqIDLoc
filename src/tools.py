import networkx as nx

def pairwise(iterable):
    import itertools
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return itertools.izip(a, b)

def outside(x, y, bounding_box):
    x0, x1, y0, y1 = bounding_box
    return x < x0 or x > x1 or y < y0 or y > y1

def construct_graph(A):
    graph = nx.DiGraph()
    for c1 in A.iterkeys():
        for c2 in A[c1].iterkeys():
            graph.add_edge(c1, c2, weight=1.0-A[c1][c2])
    return graph