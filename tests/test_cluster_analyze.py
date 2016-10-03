import numpy as np
from rtl433 import cluster_analyze


tolerance = 0.2

pulses0 = [10,10,12,20,21,22,30,31,32]
c0_3 = [[10,10,12],[20,21,22],[30,31,32]]
c0_2 = c0_3[:2]
c0_1 = c0_3[:1]

pulses1 = [1,10,10,10,100]
c1_3 = [[1],[10,10,10],[100]]
c1_2 = c1_3[:2]
c1_1 = c1_3[:1]

pulses2 = 20 * [999]
c2_3 = [pulses2]
c2_2 = [pulses2]
c2_1 = [pulses2]



def wrap(pulses, ans, max_clusters):
    anal,num_clustered = cluster_analyze(pulses, tolerance = tolerance, max_clusters=max_clusters)
    s = 0
    for a,r in zip(anal, ans):
        s += len(r)
        assert np.all(a==r)
    assert s==num_clustered

def test_generator0():

    args0 = [(pulses0, c0_3, 10), 
             (pulses0, c0_3, 3),
             (pulses0, c0_2, 2),
             (pulses0, c0_1, 1)]

    args1 = [(pulses1, c1_3, 10), 
             (pulses1, c1_3, 3),
             (pulses1, c1_2, 2),
             (pulses1, c1_1, 1)]

    args2 = [(pulses2, c2_3, 10), 
             (pulses2, c2_3, 3),
             (pulses2, c2_2, 2),
             (pulses2, c2_1, 1)]

    for arg in args0 + args1 + args2:
        yield (wrap,) + arg
