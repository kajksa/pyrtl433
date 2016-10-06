#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys
import signal
import time


import numpy as np
import matplotlib.pylab as plt


from rtl433 import RFSignal, ChuangoDemodulate, ProoveDemodulate #, OregonDemodulate


def signal_handler(signal, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

sample_rate = 250000 # TODO: Hardcoded
demodulators = [ChuangoDemodulate(), ProoveDemodulate()]

class SignalProcess(RFSignal):
    def __init__(self):
        self.dump_raw = None
        self.analyze_signal = False
        self.demodulators = []
        self.pulse_detect = True
        self.max_pulses = 500

    def initialize(self, num_samples):
        super().__init__(num_samples)

    def _pulse_detect(self):
        if len(self.pulses)>self.max_pulses:
            return False
        return True
        
    def run(self, samples):
        # Dumb everything as one stream
        if self.dump_raw:
            d.tofile(self.dump_raw)
            self.dump_raw.flush()

        self.process(samples)
        if self.analyze_signal:
            self.analyze()

        #
        # PULSE DETECT ON QUANTIZED SIGNAL
        #

        self.pulse_detected = self._pulse_detect()
        
        #
        # DEMODULATE BITS
        #

        if self.pulse_detected:
            for demod in self.demodulators:
                data = demod(self)
                demod.print()
        return
                
    def callback(self, samples, rtl):
        try:
            d = np.frombuffer(samples, dtype=np.uint8)
            self.run(d)
        except Exception as err:
            print("Error in callback!")
            print(err)
            print("sys.exit(-1)")
            #rtl.cancel_read_async()
            #rtl.close()
            #raise err
            sys.exit(-1)

parser = argparse.ArgumentParser()

# Reading raw data
parser.add_argument("-r", default=None, help="read raw data from file")
parser.add_argument("--samples", type=int)
parser.add_argument("--sample-start", type=int, default=0)
parser.add_argument("--plot", action="store_true", help="plot signal")
parser.add_argument("--saveplot", help="save plot to file")

# Options
parser.add_argument("-A", action="store_true", help="analyze signal")

# Real time
parser.add_argument("--dump-raw", help="dump raw data")
parser.add_argument("--debug-pulse-detect", help="Dumb data for debugging pulse detection")
parser.add_argument("--gain", help="RTL-SDR gain", default="auto")


args = parser.parse_args()

sp = SignalProcess()

sp.analyze_signal = args.A
if args.dump_raw:
    sp.dump_raw = open(args.dump_raw, "wb")
sp.demodulators = demodulators


# Load file recorded
if args.r:
    print("Loading file {}".format(args.r))
    d = np.fromfile(args.r, dtype=np.uint8)

    start = args.sample_start
    end = len(d)//2
    if args.samples:
        end = start + args.samples

    if 2*end>len(d):
        end = len(d)//2

    num_samples = end - start

    d = d[2*start:2*end]
    print("Loaded {} samples ({:2f} s @ {} Hz)".format(num_samples, num_samples/sample_rate, sample_rate))

    sp.initialize(num_samples)
    sp.run(d)

    if args.plot or args.saveplot:
        plt.switch_backend('Qt4Agg')
        plt.figure("data")
        x = np.arange(start,start+num_samples)
        plt.plot(x, sp.squared, 'b', label="raw")
        plt.plot(x, sp.signal, 'g', label="lowpass")
        plt.plot(x, sp.quantized, 'r', label="quantized")
        plt.xlabel("Samples")


        plt.title(args.r)

        if args.saveplot:
            plt.savefig(args.saveplot)

        if args.plot:
            plt.show()

# Realtime
else:
    from rtlsdr import RtlSdr

    freq0 = 433.92
    freq_correction = 0
    num_samples = 4*256*256*2

    try:
        gain = int(args.gain)
    except:
        gain = args.gain

    sdr = RtlSdr()
    sdr.sample_rate = sample_rate
    sdr.center_freq = freq0 * 1e6
    if freq_correction!= 0:
        sdr.freq_correction = freq_correction
    sdr.gain = gain
    
    print("RTL: Sample rate: {:.2f}".format(sdr.sample_rate))
    print("RTL: Gain: {}".format(sdr.gain))
    print("RTL: Reading samples {:.2f} s".format(num_samples/2/sdr.sample_rate))

    sp.initialize(num_samples//2)
    while True:
        sdr.read_bytes_async(sp.callback, num_samples) # Nothing is raised when callback raises errors!
