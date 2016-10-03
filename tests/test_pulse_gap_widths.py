import numpy as np
from rtl433 import pulse_gap_widths

def test0():
    d = np.array(5*[0] + 5*[1] + 1*[0] + 1*[1])
    
    w = pulse_gap_widths(d)
    assert np.all(w==[5,5,1,1])

    # Invert
    d = np.array([1 if x==0 else 0 for x in d])
    w = pulse_gap_widths(d)
    assert np.all(w==[5,5,1,1])

def test_constant():
    d = np.ones(10,dtype=np.uint8)
    w = pulse_gap_widths(d)
    assert np.all(w==[len(d)])

    d = np.zeros(10,dtype=np.uint8)
    w = pulse_gap_widths(d)
    assert np.all(w==[len(d)])

def test_fliping():
    d = np.ones(10,dtype=np.uint8)
    d[::2] = 0
    w = pulse_gap_widths(d)
    assert np.all(w==np.ones(len(d)))

    d = np.zeros(10,dtype=np.uint8)
    w = pulse_gap_widths(d)
    assert np.all(w==[len(d)])

def test_single():
    d = [0]
    w = pulse_gap_widths(d)
    assert np.all(w==[1])

    d = [1]
    w = pulse_gap_widths(d)
    assert np.all(w==[1])


