#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import sys
import signal
import time

import numpy as np
import matplotlib.pylab as plt

from rtl433 import RFSignal, ChuangoDemodulate, ProoveDemodulate

def signal_handler(signal, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

sample_rate = 250000 # TODO: Hardcoded
demodulators = [ChuangoDemodulate, ProoveDemodulate]

class SignalProcess(RFSignal):
    def __init__(self):
        self.dump_raw = None
        self.analyze_signal = False
        self.demodulators = []
        self.max_pulses = 500

    def initialize(self, num_samples):
        super().__init__(num_samples)

    def pulse_detect(self):
        #print(len(self.pulses), np.min(self.pulses), np.max(self.pulses), np.min(self.gaps), np.max(self.gaps))
        self.pulse_detected = True
        if len(self.pulses)>self.max_pulses:
            self.pulse_detected = False
            return
        if np.min(self.pulses)==1 and np.min(self.gaps):
            self.pulse_detected = False
            return
        
    def demodulate(self):
        if self.pulse_detected:
            for Demodulator in self.demodulators:
                dm = Demodulator()
                data = dm(self)
                if data:
                    print("Demodulator: {}".format(dm.name))
                    dm.print()
        
    def run(self, samples):
        # Dumb everything as one stream
        if self.dump_raw:
            d.tofile(self.dump_raw)
            self.dump_raw.flush()

        self.process(samples)
        if self.analyze_signal:
            self.analyze()
        self.pulse_detect()        
        self.demodulate()
                    
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

parser.add_argument("--port", help="port number", default=1235, type=int)
parser.add_argument("--hostname", help="hostname", default="127.0.0.1")
parser.add_argument("--client", help="TCP client", action="store_true")


args = parser.parse_args()

sp = SignalProcess()

sp.analyze_signal = args.A
if args.dump_raw:
    sp.dump_raw = open(args.dump_raw, "wb")
sp.demodulators = demodulators

# Load recorded file
if args.r:
    print("Loading file {}".format(args.r))
    d = np.fromfile(args.r, dtype=np.uint8)

    start = args.sample_start
    assert start % 2 == 0
    end = len(d)
    assert end %2 == 0

    if args.samples:
        assert args.samples % 2 == 0
        end = start + args.samples

    if end>len(d):
        end = len(d)

    num_samples = end - start
    d = d[start:end]
    
    print("Loaded {} samples ({:2f} s @ {} Hz)".format(num_samples, num_samples/2/sample_rate, sample_rate))

    sp.initialize(num_samples)
    sp.run(d)

    if args.plot or args.saveplot:
        plt.switch_backend('Qt4Agg')
        plt.figure("data")
        x = np.arange(start,start+num_samples, 2)
        plt.plot(x, sp.squared, 'b', label="squared")
        plt.plot(x, sp.signal, 'g', label="lowpass")
        plt.plot(x, sp.quantized, 'r', label="quantized")
        plt.xlabel("Samples")
        plt.title(args.r)
        if args.saveplot:
            plt.savefig(args.saveplot)
        if args.plot:
            plt.show()

# RTL dongle/tcp client
else:
    from rtlsdr import RtlSdr, RtlSdrTcpClient

    # Monkey patch to avoid print statement, can not get _keep_alive to work?
    def _close_socket(self):
        if self._keep_alive:
            return
        s = getattr(self, '_socket', None)
        if s is None:
            return
        #print('client closing socket')
        s.close()
        self._socket = None
    RtlSdrTcpClient._close_socket = _close_socket
        
    freq0 = 433.92
    freq_correction = 0
    num_samples = 4*256*256*2

    try:
        gain = int(args.gain)
    except:
        gain = args.gain

    if args.client==False:
        sdr = RtlSdr()
    else:
        sdr = RtlSdrTcpClient(hostname=args.hostname, port=args.port)
        
    sdr.sample_rate = sample_rate
    sdr.center_freq = freq0 * 1e6
    if freq_correction!= 0:
        sdr.freq_correction = freq_correction
    sdr.gain = gain
    
    print("RTL: Sample rate: {:.2f}".format(sdr.sample_rate))
    print("RTL: Gain: {}".format(sdr.gain))
    print("RTL: Reading samples {:.2f} s".format(num_samples/2/sdr.sample_rate))

    sp.initialize(num_samples)
    if args.client==False:
        while True:
            sdr.read_bytes_async(sp.callback, num_samples) # Nothing is raised when callback raises errors!
    else:
        # sdr._keep_alive = True
        while True:
            samples = sdr.read_bytes(num_samples)
            samples = np.array(samples, dtype=np.uint8)
            sp.run(samples)
