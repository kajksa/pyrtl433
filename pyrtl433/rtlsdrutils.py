import json

import numpy as np

import crtlsdrutils

# http://stackoverflow.com/questions/17479944/partitioning-an-float-array-into-similar-segments-clustering
# http://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-450-principles-of-digital-communications-i-fall-2006/lecture-notes/book_3.pdf
def lloyd_max(x, xq0, max_iter = 5):
    """Lloyd-Max algorithm"""
    n = len(xq0)
    xq = xq0 + 0

    err_old = 1e10
    iteration = 0
    while True:
        tq = (xq[:-1] + xq[1:])//2
        # TODO:    http://stackoverflow.com/questions/9444409/why-is-numpy-much-slower-than-matlab-on-a-digitize-example
        b = np.digitize(x,tq)
        err = np.var(x-xq[b])
        print(err_old, err, iteration, xq)
        if err_old==0 or np.abs(err-err_old)/err_old < 1e-3 or iteration>=max_iter:
            return xq, b
        err_old = err

        for i in range(n): # Update levels
            d = x[b == i]
            if len(d)>0:
                xq[i] = np.mean(d)
        iteration += 1

def lloyd_max_bin(x, xq0, max_iter = 5):
    """Simplified Lloyd-Max for low, high data (two-levels)."""
    assert len(xq0)==2
    xq = xq0 + 0

    N = len(x)
    iteration = 0
    b = np.zeros_like(x)
    while True:
        tq = (xq[0] + xq[1])//2
                
        I = x>tq

        b[:] = 0
        b[I] = 1

        if iteration>=max_iter:
            return xq, b

        s = np.sum(I)
        if s!=0:
            xq[1] = np.sum(x[I])//s

        if s!=N:
            xq[0] = np.sum(x[~I])//(N-s)

        # TODO: when x[I] or x[~I] is empty? Should not happen when xq0 = [min, max], BUT expensive
        # xq[0] = np.mean(x[~I])
        # xq[1] = np.mean(x[I])
        
        iteration += 1
    



def _calc_uint8_squares():
    """Lookup table of centered squared np.uint8 stored as np.uint16."""
    s = np.empty(256, dtype=np.uint16)
    for i in range(256):
        s[i] = (127 - i) * (127 - i)
    return s

# Lookup table
_uint8_squares = _calc_uint8_squares()

def square_uint8(d):
    """Find squares of centered uint8 data using look up table."""
    assert d.dtype==np.uint8
    return _uint8_squares[d[::2]] + _uint8_squares[d[1::2]]


def pulse_gap_widths(b):
    """From a sequnce of levels 0 and 1 levels return widths of constant levels."""
    #ispulse = np.concatenate(([b[0]], np.equal(b, 1).view(np.int8), [0 if b[-1]==1 else 1]))
    ispulse = np.empty(len(b) + 2, dtype=np.int8)
    ispulse[0] = b[0]
    ispulse[1:-1] = b
    ispulse[-1] = 0 if b[-1]==1 else 1
    absdiff = np.abs(np.diff(ispulse))
    ranges = np.concatenate(([0],np.where(absdiff == 1)[0])) # TODO: remove concatenate
    widths = np.diff(ranges)
    return widths

def split_on_gaps(widths, start_value, threshold):
    """Split widths where widths[0] has start_value (0: low, gap, 1: high, pulse)."""
    assert start_value in [0,1] # low, high
    if start_value == 0:
        gaps = widths[0::2]
        pulses = widths[1::2]
    else:
        gaps = widths[1::2]
        pulses = widths[0::2]

    where_reset = np.where(gaps>threshold)[0]
    pulsess = np.split(pulses,where_reset+start_value)
    gapss = np.split(gaps,where_reset+1)

    assert len(pulsess)==len(gapss)
    
    return pulsess, gapss

def split_packet(pulses, gaps, reset_limit):
    """Simplified split_on_gaps, assuming starting on pulse and ending on gap."""
    where_reset = np.where(gaps>reset_limit)[0]
    pulsess = np.split(pulses,where_reset+1)
    gapss = np.split(gaps,where_reset+1)
    assert len(pulsess)==len(gapss)
    return pulsess, gapss



# http://stackoverflow.com/questions/17479944/partitioning-an-float-array-into-similar-segments-clustering
def cluster_analyze(L, tolerance, max_clusters = 10):
    """Simple 1D cluster analysis."""
    f = int(tolerance * 100) + 100 # 0.2 -> 120
    Ls = np.sort(L) # Expensive
    clusters = [0]
    current = Ls[0]
    for i, l in enumerate(Ls):
        if l > (f * current) // 100: # L is int
            clusters.append(None)
        clusters[-1] = i + 1
        current = l
        if len(clusters) == max_clusters + 1:
            break

    num_clustered = i + 1 # Number of points clustered
    clustdata = np.split(Ls, clusters[:-1])
    if len(clusters) == max_clusters + 1:
        num_clustered = i
        clustdata = clustdata[:-1]

    assert len(clustdata)<=max_clusters

    return clustdata, num_clustered 


def cluster_median(c):
    """Find median off sorted sequence c."""
    Nc = len(c)
    if Nc % 2 == 0:
        m = Nc // 2
    else:
        m = (Nc - 1) // 2
    return c[m]

def pack_bytes(bits_bool):
    """Pack array of bools as bytes (np.uint8)."""
    bits_bool = bits_bool.astype(int)
    bytes = np.packbits(bits_bool)
    return bytes
                    

class RFSignal:
    def __init__(self, num_samples):
        self.num_samples = num_samples
        self.num_bytes = 2 * self.num_samples

        # TODO
        # Numerator (b) and denominator (a) polynomials of the IIR filter
        #b, a = scipy.signal.butter(1, 0.05) # Digital filter
        b = [ 0.07295966, 0.07295966]
        a = [ 1., -0.85408069] # Notice the minus! Needed in fixed point lowpass stuff
        self.bi,self.ai = crtlsdrutils.lowpass_params(b,a)

        #
        # Pre allocate
        #

        # Squared signal
        self.squared = np.empty(self.num_samples, dtype = np.uint16)
        # Low pass filtered signal
        self.signal = np.empty(self.num_samples, dtype = np.uint16)


    def _square(self):
        crtlsdrutils.square_uint8(self.data, self.squared)

    def _lowpass(self):
        crtlsdrutils.lowpass_uint16(self.bi, self.ai, self.squared, self.signal, state=np.array([self.squared[0], self.squared[0]]))
                        

    def _quantize(self):
        # TODO! This is slow!
        #xq0 = np.array([np.min(self.signal), np.max(self.signal)]) # How fast is this? Many reductions!
        xq0 = np.array([0, np.max(self.signal)], dtype = np.uint16) # Faster and more robust

        #self.levels, self.level_index = lloyd_max_bin(self.signal, xq0=xq0, max_iter=1)
        self.levels, self.level_index = crtlsdrutils.lloyd_max_bin(self.signal, xq0=xq0, max_iter=1)
        self.start_low = True if self.level_index[0]==0 else False
        self.end_low = True if self.level_index[-1]==0 else False

    def _pulses_gaps(self):
        # Find widths of gaps and pulse
        self.widths = crtlsdrutils.pulse_gap_widths(self.level_index)
        
        # Always start on a pulse and end on a gap
        if self.start_low:
            self.widths = self.widths[1:]
        if not self.end_low:
            self.widths = self.widths[:-1]
        assert len(self.widths) % 2 == 0
        
        self.pulses = self.widths[0::2]
        self.gaps = self.widths[1::2]

        self.num_pulses = len(self.pulses)

    def process(self, data):
        self.data = data
        assert len(data) == self.num_bytes
        assert data.dtype == np.uint8

        # Square data
        self._square()

        # Lowpass filter squared
        self._lowpass()
        
        # Quantize signal
        self._quantize()

        # Find pulses and gaps
        self._pulses_gaps()


    @property
    def quantized(self):
        return self.levels[self.level_index]

    def analyze(self, max_clusters=16, tolerance=0.2, verbose=True):

        periods = self.pulses + self.gaps

        # Cluster widths of pulses, gaps and periods
        cpulses, _ = cluster_analyze(self.pulses, tolerance = tolerance, max_clusters=max_clusters)
        cgaps, _ = cluster_analyze(self.gaps, tolerance = tolerance, max_clusters=max_clusters)
        cperiods, _ = cluster_analyze(periods, tolerance = tolerance, max_clusters=max_clusters)

        if verbose:
            print("Analyzing pulses...")
            print("Total count: {},  width: {}".format(self.num_pulses, np.sum(periods)))

            def _print(cc):
                for clusterid,c in enumerate(cc):
                    data = {'clusterid': clusterid, 'count': len(c), 'width': cluster_median(c), 'min': c[0], 'max': c[-1]}
                    print("[{clusterid}] count: {count:>4},  width:    {width} [{min};{max}]".format(**data))
                
            print("Pulse width distribution:")
            _print(cpulses)
            print("Gap width distribution:")
            _print(cgaps)
            print("Pulse period distribution:")
            _print(cperiods)
            print()

            print("Guessing modulation:")
            if len(cpulses)==1:
                print("Pulse Position Modulation with fixed pulse width")
            elif len(cperiods)==2:
                print("Pulse Width Modulation with fixed period")
            else:
                print("No clue...")

        return cpulses, cgaps, cperiods



def bytes2str(bs):
    """Convert iterable of bytes to hex integer string for printing."""
    ret = ""
    for x in bs:
        ret += "{:02x} ".format(x)
    return ret[:-1]

def boolbit2str(boolbit):
    """Convert boolean array to bit string for printing."""
    ret = ""
    zero = "0"
    one = "1"
    for i in range(len(boolbit)//8):
        byte = boolbit[i*8:(i+1)*8]
        for b in byte:
            if b:
                ret += one
            else:
                ret += zero
        ret += " "
    byte = boolbit[8*(len(boolbit)//8):]
    for b in byte:
        if b:
            ret += one
        else:
            ret += zero
    return ret


#
# DEMODULATE
#

# Proove: PPM with fixed pulse width
# Pulse 62 +- 3
# Gaps 

# Chuango: PWM with fixed period
# File: chuango_basementdoor/gfile014.data
# Guessing modulation: Pulse Width Modulation with fixed period
# Attempting demodulation... short_limit: 267, long_limit: 398, reset_limit: 398


# Working with nibbles
#define HI_NIBBLE(b) (((b) >> 4) & 0x0F)
#define LO_NIBBLE(b) ((b) & 0x0F)

# https://wiki.python.org/moin/BitManipulation

def manchester(bits):
    """Manchester decoding."""
    if len(bits) % 2!= 0 or np.any(bits[::2]==bits[1::2]):
        raise ValueError("Not valid data for Manchester decoding.")
    return bits[::2]

class Demodulate:
    def _print(self, i, bytes, boolbit):
        # Print
        sb = bytes2str(bytes)
        sbb = boolbit2str(boolbit)
        str_pulse_analyze = "[{:02}] {{{:02}}} : {}: {}".format(i, len(boolbit), sb, sbb)
        if len(str_pulse_analyze)>80:
            s = str_pulse_analyze[:76] + " ..."
            assert len(s)==80
            print(s)
        else:
            print(str_pulse_analyze)
    def _print_data(self, i, data):
        if data:
            json_data = json.dumps(data)
            print("[{:02}] : {}".format(i, json_data))

    def print(self):
        for i, (boolbit, bytes, pdata) in enumerate(zip(self.boolbits, self.bytess, self.data)):
            if not np.all(boolbit==False) and len(boolbit)>2:
                self._print(i, bytes, boolbit)
                self._print_data(i, pdata)

            # if not np.all(boolbit==False):
            #     self._print(i, bytes, boolbit)
            #     self._print_data(i, pdata)
            
    def _split_packet(self, rf):
        pulsess, gapss = split_packet(rf.pulses, rf.gaps, self.reset_limit)
        return pulsess, gapss 


    
class ChuangoDemodulate(Demodulate):
    def __init__(self):
        self.reset_limit = 3800
        self.short_limit = 200

    def __call__(self, rf):
        pulsess, gapss = self._split_packet(rf)

        self.boolbits = []
        for pulses,gaps in zip(pulsess, gapss):
            if len(pulses)>0:
                # Some demodulation! Converting some series of pulses and gaps to raw bytes
                boolbit = pulses<200 # More robust?
                boolbit = ~boolbit # Long pulses are 1's in PWM ?
                self.boolbits.append(boolbit)

        self.bytess = [pack_bytes(b) for b in self.boolbits]

        self.data = []
        for i, (boolbit, bytes) in enumerate(zip(self.boolbits, self.bytess)):
            pdata = {}
            # 25 bits, always ending on a short pulse, and not all device id bits equal
            if len(boolbit)==25 and not boolbit[-1] and not np.all(boolbit[0]==boolbit[0:20]):
                pdata["device_id"] = int((bytes[0] << 12) | (bytes[1] << 4) | (bytes[2] >> 4))
                pdata["cmd_id"] = int(bytes[2] & 0x0F)
                pdata["product"] = "Chuango"
            self.data.append(pdata)
        return self.data

class ProoveDemodulate(Demodulate):
    def __init__(self):
        self.reset_limit = 2400
        self.short_limit = 100

    def __call__(self, rf):
        pulsess, gapss = self._split_packet(rf)

        self.boolbits = []
        for pulses,gaps in zip(pulsess, gapss):
            if len(pulses)>0:
                # Some demodulation! Converting some series of pulses and gaps to raw bytes
                boolbit = gaps<self.short_limit
                try:
                    boolbit = manchester(boolbit[1:-1])
                    self.boolbits.append(boolbit)
                except ValueError:
                    continue

        self.bytess = [pack_bytes(b) for b in self.boolbits]
        
        self.data = []
        for i, (boolbit, bytes) in enumerate(zip(self.boolbits, self.bytess)):
            pdata = {}            
            if len(boolbit)==32:
                pdata["id"] = int((bytes[0] << 18) | (bytes[1] << 10) | (bytes[2] << 2) | (bytes[3]>>6)) #  ID 26 bits
                pdata["group"] = int((bytes[3] >> 5) & 1) 
                pdata["state"] = "OFF" if ((bytes[3] >> 4) & 1)==1 else "ON"
                pdata["channel"] = int((bytes[3] >> 2) & 0x03)
                pdata["unit"] = int((bytes[3] & 0x03))
                pdata["product"] = "Proove"
            self.data.append(pdata)
        return self.data


# class OregonDemodulate(Demodulate):
#     def __init__(self):
#         self.reset_limit = 2400
#         self.short_limit = 130

#     def __call__(self, rf):
#         pulsess, gapss = self._split_packet(rf)

#         self.boolbits = []
#         for pulses,gaps in zip(pulsess, gapss):
#             if len(pulses)>0:
#                 # Some demodulation! Converting some series of pulses and gaps to raw bytes
#                 boolbit = (pulses<self.short_limit)
#                 boolbit2 = (gaps<self.short_limit)
#                 print("Pulse short", boolbit)
#                 print("Gaps short", boolbit2)
#                 try:
#                     boolbit = manchester(boolbit)
#                     self.boolbits.append(boolbit)
#                 except ValueError:
#                     continue

#         self.bytess = [pack_bytes(b) for b in self.boolbits]
        
#         print(self.boolbits)
        
#         #return ldata




if __name__ == "__main__":

    import matplotlib.pylab as plt
    
    N = 100

    x = np.zeros(N)
    x[1:5] = 1
    x[10:20] = 1
    x[40:50] = 1
    x[55:56] = 1.2
    x[90:100] = 1
    x = x + 0.2*(np.random.random(N)-0.5)

    # Convert signal to uint8
    x = x * 255
    x[x>255] = 255
    x[x<0] = 0
    x = np.asarray(x, dtype=np.uint8)

    xq0 = np.array([10,11], dtype=np.uint8)
    #xq0 = None
    max_iter = 10

    xq, b = lloyd_max_bin(x,xq0=xq0, max_iter=max_iter)

    print("Result: levels = ", xq)
    print("Result: bits = ", b)

    plt.figure("result")
    plt.plot(x)
    plt.plot(xq[b], "o")
    plt.show()
