import numpy as np

from rtl433 import split_on_gaps

reset_limit = 100

#
# TODO
#


start_gap = 3000
end_gap = 2000
pulse_length = 10
gap_length = 30
packet_gap = 200


one_pulses = [pulse_length] * 3
one_gaps = [gap_length] * 3

# Create one data packet
one = [start_gap]
for p,g in zip(one_pulses, one_gaps):
    one += [p,g]
one = one[:-1] + [end_gap]

# Three identical packets
three = one[:-1] + [packet_gap] + one[1:-1] + [packet_gap] + one[1:]

# start_value = 0
one = np.array(one)
three = np.array(three)

# start_value = 1
one_high = one[1:]
three_high = three[1:]

from itertools import zip_longest
def _reconstruct(pulsess, gapss):
    r = []
    for pulses,gaps in zip(pulsess, gapss):
        for p,g in zip_longest(pulses,gaps):
            if p is not None:
                r.append(p)
            if g is not None:
                r.append(g)
    return r
            
def test_one_low_low():
    pulsess, gapss = split_on_gaps(one, 0, 100)
    r = _reconstruct(pulsess, gapss)
    assert np.all(r==one)

def test_three_low_low():
    pulsess, gapss = split_on_gaps(three, 0, 100)
    r = _reconstruct(pulsess, gapss)
    assert np.all(r==three)

def test_three_low_high():
    pulsess, gapss = split_on_gaps(three[:-1], 0, 100)
    r = _reconstruct(pulsess, gapss)
    assert np.all(r==three[:-1])

def test_three_high_low():
    pulsess, gapss = split_on_gaps(three_high, 1, 100)
    r = _reconstruct(pulsess, gapss)
    assert np.all(r==three_high)

def test_three_high_high():
    pulsess, gapss = split_on_gaps(three_high[:-1], 1, 100)
    r = _reconstruct(pulsess, gapss)
    assert np.all(r==three_high[:-1])

