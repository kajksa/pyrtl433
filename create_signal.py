# -*- coding: utf-8 -*-
import sys
import time

import numpy as np
import scipy.signal


from rtl433 import RFSignal, cluster_median

filename = sys.argv[1]
fileoutname = sys.argv[2]

print("Loading file {}".format(filename))

d = np.fromfile(filename, dtype=np.uint8)

num_samples = len(d)//2

print("Loaded {} samples".format(num_samples))

rf = RFSignal(num_samples)

rf.process(d)

assert rf.start_low==True, "Input signal should start with a gap"
assert rf.end_low==True, "Input signal should end with a gap"

cpulses, cgaps, cperiods = rf.analyze()

# Create normalized width signal starting with a pulse
signal = np.empty(len(rf.widths), dtype=np.uint16)
low = False
for i,w in enumerate(rf.widths):
    wn = -1
    if low:
        for c in cgaps:
            if w in c:
                wn = cluster_median(c)
                break
        low = False
    else:
        for c in cpulses:
            if w in c:
                wn = cluster_median(c)
                break
        low = True
    assert wn>0
    signal[i] = wn

print("Width normalized signal:", signal)

# "Normalized" pulses and gaps
pulses = signal[::2]
gaps = signal[1::2]
assert len(pulses)==len(gaps)

print("Pulse widths", set(pulses))
print("Gap widths", set(gaps))

print("Saving file: {}".format(fileoutname))
np.save(fileoutname, signal)
print("DONE")
