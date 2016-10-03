import numpy as np
from rtlsdrutils import lloyd_max, lloyd_max_bin

N = 100
pulse_value = 17
gap_value = 1


# 1 state
signal0 = gap_value * np.ones(N, dtype=np.int)
answer0_b = np.array(signal0==pulse_value, dtype=np.int)
answer0_xq = [gap_value, pulse_value]

# 2 states
signal1 = gap_value * np.ones(N, dtype=np.int)
signal1[20:30] = pulse_value
signal1[50:60] = pulse_value
answer1_b = np.array(signal1==pulse_value, dtype=np.int)
answer1_xq = [gap_value, pulse_value]


# 3 states
pulse_value1 = 10
signal2 = gap_value * np.ones(N, dtype=np.int)
signal2[20:30] = pulse_value
signal2[50:60] = pulse_value
signal2[65:70] = pulse_value1
signal2[90:93] = pulse_value1
answer2_b = np.array(signal2==pulse_value1, dtype=np.int)
answer2_b[signal2==pulse_value] = 2
answer2_xq = [gap_value, pulse_value1, pulse_value]


def test_signal0():
    max_iter = 5

    xq0 = np.array([gap_value+1, pulse_value], dtype=np.int)
    xq, b = lloyd_max_bin(signal0, xq0, max_iter = max_iter)
    assert np.all(xq==answer0_xq)
    assert np.all(b==answer0_b)

    xq, b = lloyd_max(signal0, xq0, max_iter = max_iter)
    assert np.all(xq==answer0_xq)
    assert np.all(b==answer0_b)


def test_signal1():
    max_iter = 5

    xq0 = np.array([gap_value+1, pulse_value-1], dtype=np.int)
    xq, b = lloyd_max_bin(signal1, xq0, max_iter = max_iter)    
    assert np.all(xq==answer1_xq)
    assert np.all(b==answer1_b)

    xq0 = np.array([gap_value+1, pulse_value-1], dtype=np.int)
    xq, b = lloyd_max(signal1, xq0, max_iter = max_iter)    
    assert np.all(xq==answer1_xq)
    assert np.all(b==answer1_b)


def test_signal2():
    max_iter = 5

    xq0 = np.array([gap_value+1, pulse_value1+1, pulse_value-1], dtype=np.int)
    xq, b = lloyd_max(signal2, xq0, max_iter = max_iter)    
    assert np.all(xq==answer2_xq)
    assert np.all(b==answer2_b)

