import sys
import numpy as np
cimport numpy as np
cimport cython


# #define F_SCALE 15
# #define S_CONST (1<<F_SCALE)
# #define FIX(x) ((int)(x*S_CONST))
# ///  [b,a] = butter(1, 0.01) -> 3x tau (95%) ~100 samples
# //static int a[FILTER_ORDER + 1] = {FIX(1.00000), FIX(0.96907)};
# //static int b[FILTER_ORDER + 1] = {FIX(0.015466), FIX(0.015466)};
# ///  [b,a] = butter(1, 0.05) -> 3x tau (95%) ~20 samples
# static int a[FILTER_ORDER + 1] = {FIX(1.00000), FIX(0.85408)};
# static int b[FILTER_ORDER + 1] = {FIX(0.07296), FIX(0.07296)};


cdef int F_SCALE = 15
cdef int S_CONST = 1 << F_SCALE

cdef int one = 1

cdef FIX(x):
    return int(x*S_CONST)

def lowpass_params(b,a):
    assert len(b)==len(a)==2
    a = np.array([FIX(a[0]), FIX(-a[1])], dtype=np.uint16)
    b = np.array([FIX(b[0]), FIX(b[1])], dtype=np.uint16)
    return b, a


@cython.wraparound(False)
@cython.boundscheck(False)
def lowpass_uint16(np.ndarray[np.uint16_t, ndim=1] b, np.ndarray[np.uint16_t, ndim=1] a, np.ndarray[np.uint16_t, ndim=1] x, np.ndarray[np.uint16_t, ndim=1] y, np.ndarray[np.uint16_t, ndim=1] state):
    """First order IIR filter with fixed point arithmetic."""

    cdef int i
    cdef int N = x.shape[0]

    y[0] = ((a[1] * state[1] >> 1) + (b[0] * x[0] >> 1) + (b[1] * state[0] >> 1)) >> (F_SCALE - 1)
    for i in range(1,N):
        y[i] = ((a[1] * y[i - 1] >> 1) + (b[0] * x[i] >> 1) + (b[1] * x[i - 1] >> 1)) >> (F_SCALE - 1)

    state[0] = x[-1]
    state[1] = y[-1]


# Lookup table
def _calc_uint8_squares():
    s = np.empty(256, dtype=np.uint16)
    for i in range(256):
        s[i] = (127 - i) * (127 - i)
    return s

squares = _calc_uint8_squares()

@cython.wraparound(False)
@cython.boundscheck(False)
def square_uint8(np.ndarray[np.uint8_t, ndim=1] x, np.ndarray[np.uint16_t, ndim=1] y):
    cdef int i
    cdef int N = x.shape[0]
    cdef np.ndarray[np.uint16_t, ndim=1] p = squares

    for i in range(N//2):
        #y[i] = squares[x[2*i]] + squares[x[2*i+1]]
        y[i] = p[x[2*i]] + p[x[2*i+1]]

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def lloyd_max_bin(np.ndarray[np.uint16_t, ndim=1] x, np.ndarray[np.uint16_t, ndim=1] xq0, int max_iter = 5):
    """Simplified lloyd-max for low, high data."""
    assert len(xq0)==2
    cdef np.ndarray[np.uint16_t, ndim=1] xq = xq0 + 0

    cdef int N = len(x)
    cdef int iteration = 0
    cdef np.ndarray[np.uint16_t, ndim=1] b = np.zeros_like(x)
    cdef np.uint32_t tq
    cdef np.uint32_t s
    cdef np.uint32_t high_sum
    cdef np.uint32_t low_sum

    while True:
        tq = (xq[0] + xq[1])/2

        s = 0
        high_sum = 0
        low_sum = 0
        for i in range(N):
            if x[i]>tq:
                b[i] = 1
                s += 1
                high_sum += x[i]
            else:
                b[i] = 0
                low_sum += x[i]
        if iteration>=max_iter:
            return xq, b

        if s!=0:
            xq[1] = high_sum/s
        if s!=N:
            xq[0] = low_sum/(N-s)

        iteration += 1

@cython.wraparound(False)
@cython.boundscheck(False)
def pulse_gap_widths(np.ndarray[np.uint16_t, ndim=1] b):
    
    cdef int N = len(b)
    cdef int last = b[0]
    cdef int count = 1
    cdef int j = 0
    cdef np.ndarray[np.uint16_t, ndim=1] widths = np.empty_like(b)
    for i in range(1,N):
        if last!=b[i]:
            widths[j] = count
            count = 1
            last = b[i]
            j += 1
        else:
            count +=1

    if last==b[N-1]:
        widths[j] = count
        j += 1
        
    return widths[:j]
    
