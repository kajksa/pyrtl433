import time
import asyncio
import datetime
import argparse

import numpy as np
from rtlsdr import RtlSdr

from rtlsdrutils import RFSignal, ChuangoDemodulate, ProoveDemodulate


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


async def streaming():
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
    
    async for samples in sdr.stream(num_samples, format='bytes'):
        sample_count += 1
        #print(sample_count)

        # Need to handle overlap? Is samples save to use?    
        d = np.array(samples)

        # Dumb everything as one stream
        if args.dump_raw:
            d.tofile(fdump)
            
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
            continue
        
        #
        # DEMODULATE BITS
        #

        data = chuango(rf)
        #print("Chuango", data)
        data = proove(rf)
        #print("Proove", data)
        
        chuango.print()
        proove.print()
                
    # Stop streaming
    sdr.stop()
    sdr.close()

loop = asyncio.get_event_loop()
loop.run_until_complete(streaming())
