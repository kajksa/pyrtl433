#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys
import time

import numpy as np

from rtl433 import RFSignal, ChuangoDemodulate, ProoveDemodulate #, OregonDemodulate



parser = argparse.ArgumentParser()
parser.add_argument("-r", help="read raw data from file")
parser.add_argument("--samples", type=int)
parser.add_argument("--sample-start", type=int, default=0)
parser.add_argument("--plot", action="store_true", help="plot signal")
parser.add_argument("--saveplot", help="save plot to file")
args = parser.parse_args()


sample_rate = 250000
center_freq = 433.92e6

filename = args.r
print("Loading file {}".format(filename))
d = np.fromfile(filename, dtype=np.uint8)

start = args.sample_start
end = len(d)//2
if args.samples:
    end = start + args.samples

if 2*end>len(d):
    end = len(d)//2

num_samples = end - start

d = d[2*start:2*end]
print(len(d))
print("Loaded {} samples ({:2f} s @ {} Hz)".format(num_samples, num_samples/sample_rate, sample_rate))

rf = RFSignal(num_samples)


rf.process(d)
cpulses, cgaps, cperiods = rf.analyze()

#np.save("signals/long0.data.pulses",rf.pulses)
#np.save("signals/long0.data.gaps",rf.gaps)

# #
# # BIT DETECTION ON QUANTIZED SIGNAL
# #

# # http://se.mathworks.com/help/phased/ug/neyman-pearson-hypothesis-testing.html

# # Algo 1 - not good
# # Fixed level difference
# bit_level_detection = 150
# bitdetect1 = False
# level_diff = np.diff(rf.levels)
# if level_diff>bit_level_detection:
#     bitdetect1 = True
# print("Algo1: {} (level difference = {}, {})".format(bitdetect1, level_diff, bit_level_detection))

# #
# # BIT DETECTION ON PULSE/GAP WIDTHS
# #


# # # Algo 2    
# # min_pulse_width = 50
# # N = np.sum(pulses > min_pulse_width)
# # bitdetect2 = False
# # if N>0:
# #     bitdetect2 = True
# # print("Algo2: {} ({} pulse widths larger than {})".format(bitdetect2, N, min_pulse_width))

# # # Algo 3
# # max_gap_width_threshold = 2400
# # max_gap_width = np.max(gaps)
# # bitdetect3 = False
# # if max_gap_width>max_gap_width_threshold:
# #     bitdetect3 = True
# # print("Algo3: {} (max_gap_width = {}, {})".format(bitdetect3, max_gap_width, max_gap_width_threshold))

# # # Algo 4
# # q_err_ratio_threshold = 0.5 # Empirical

# # I = b==1
# # q_err = dd_low.astype(np.int32) - xq[b]
# # q_err_low = dd_low.astype(np.int32) - xq[0]
# # q_err_ratio = np.sum(np.abs(q_err[I]))/np.sum(np.abs(q_err_low[I]))
# # bitdetect4 = False
# # if q_err_ratio< q_err_ratio_threshold:
# #     bitdetect4 = True
# # print("Algo4: {} (quantize_error_ratio = {:.2f}, {})".format(bitdetect4,q_err_ratio, q_err_ratio_threshold))

# # Algo 5
# #bitdetect5 = np.any(rf.pulses[0:10]>50)
# bitdetect5 = np.any(rf.pulses>50)
# print("Algo5: {}".format(bitdetect5))

# bitdetects = np.array([bitdetect1, bitdetect5])

# if not np.all(bitdetects[0]==bitdetects):
#     print("WARNING: bit detection algorithms do not agree")

# bitdetect = bitdetect5




chuango = ChuangoDemodulate()
data = chuango(rf)

proove = ProoveDemodulate()
data = proove(rf)

chuango.print()
proove.print()


# oregon = OregonDemodulate()
# data = oregon(rf)



#
# PLOT
#



if args.plot or args.saveplot:
    import matplotlib.pylab as plt
    plt.switch_backend('Qt4Agg')
    plt.figure("data")
    x = np.arange(start,start+num_samples)
    plt.plot(x, rf.squared, 'b', label="raw")
    plt.plot(x, rf.signal, 'g', label="lowpass")
    plt.plot(x, rf.quantized, 'r', label="quantized")
    plt.xlabel("Samples")

    # if bitdetect:
    #     title = "{}: Bits".format(filename)
    # else:
    #     title = "{}: NO bits".format(filename)
    # plt.title(title)

    plt.title(filename)
        
    if args.saveplot:
        plt.savefig(args.saveplot)

    # plt.figure("freq")
    # from matplotlib.mlab import psd 

    # pxx, freqs = psd(d-np.mean(d), NFFT=2**10, Fs=sample_rate/1e6) #, Fc=center_freq/1e6)
    # plt.plot(center_freq/1e6 + freqs, 10*np.log10(pxx))
    # plt.ylim(0,70)

    # plt.figure("hist")
    # for c in cpulses:
    #     plt.hist(c)
        
    
    if args.plot:
        plt.show()

print("DONE")

