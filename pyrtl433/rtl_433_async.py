import sys
import time
import datetime
import argparse

import numpy as np
from rtlsdr import RtlSdr

from rtlsdrutils import RFSignal, ChuangoDemodulate, ProoveDemodulate


import signal


def signal_handler(signal, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

parser = argparse.ArgumentParser()
parser.add_argument("--dump-raw", help="dump raw data")
parser.add_argument("--debug-pulse-detect", help="Dumb data for debugging pulse detection")
parser.add_argument("--gain", help="RTL-SDR gain", default="auto")


args = parser.parse_args()


freq0 = 433.92

freq_correction = 0

try:
    gain = int(args.gain)
except:
    gain = args.gain

sample_rate = 250000
num_samples = 4*256*256*2

num_callbacks = 2

sdr = RtlSdr()
sdr.sample_rate = sample_rate
sdr.center_freq = freq0 * 1e6
if freq_correction!= 0:
    sdr.freq_correction = freq_correction
sdr.gain = gain

print("Sample rate: {}".format(sdr.sample_rate))
print("Gain: {}".format(sdr.gain))

print("Reading samples {} s".format(num_samples/2/sdr.sample_rate))
    
rf = RFSignal(num_samples//2)

chuango = ChuangoDemodulate()
proove = ProoveDemodulate()

fdump = None
if args.dump_raw:
    fdump = open(args.dump_raw, "wb")

debug_pd_template0 = "debug_pulse_{}.bin"
debug_pd_template1 = "debug_nopulse_{}.bin"

pulse_detect_count = 0
nopulse_detect_count = 0
sample_count = 0


from rtlsdr.helpers import limit_calls
#@limit_calls(num_callbacks)
def read_callback(samples, rtl):

        #sample_count += 1
        
        #sample_count += len(samples)
        
        #print(sample_count)

        #print(type(samples))
        #print(samples.dtype)

        
        # Need to handle overlap? Is samples save to use?    
        #d = np.array(samples)
        d = np.frombuffer(samples, dtype=np.uint8)

        # Dumb everything as one stream
        if args.dump_raw:
            d.tofile(fdump)
            fdump.flush()

        rf.process(d)
        #rf.analyze()

        #
        # PULSE DETECT ON QUANTIZED SIGNAL
        #

        # pulse_detect_max_pulses = 500
        # pulse_detect_threshold_min_num = 16
        # pulse_detect_threshold = 50
        # pulse_detected = (len(rf.pulses) < pulse_detect_max_pulses) and np.sum(rf.pulses>pulse_detect_threshold)>=pulse_detect_threshold_min_num

        # if pulse_detected:
        #     print("Pulse detected")
        #     if args.debug_pulse_detect:
        #         d.tofile(debug_pd_template0.format(pulse_detect_count))
        #     pulse_detect_count += 1
        # else:
        #     print("No pulse detected")
        #     if args.debug_pulse_detect:
        #         d.tofile(debug_pd_template1.format(nopulse_detect_count))
        #     nopulse_detect_count += 1

        # assert pulse_detect_count + nopulse_detect_count == sample_count

        # # Skip demodulation if no pulses detected
        # if pulse_detected==False:
        #     continue


        if len(rf.pulses)>500:
            return

        #
        # DEMODULATE BITS
        #

        data = chuango(rf)
        #print("Chuango", data)
        data = proove(rf)
        #print("Proove", data)

        chuango.print()
        proove.print()


# TODO: How to break clean?
while True:
    try:
        print("Calling read_bytes_async()")
        sdr.read_bytes_async(read_callback, num_samples)
        print("Done")
    except KeyboardInterrupt:
        print("DISCO"*10)
        sdr.close() # cancel_read_async()
        break
