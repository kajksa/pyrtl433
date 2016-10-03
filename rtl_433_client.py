import time
import asyncio
import datetime
import argparse

import numpy as np
from rtlsdr import RtlSdr, RtlSdrTcpClient

from rtl433 import RFSignal, ChuangoDemodulate, ProoveDemodulate

sample_rate = 250000
num_samples = 4*256*256*2
gain = 49.6


sdr = RtlSdrTcpClient(port=1235)

sdr.sample_rate = sample_rate
sdr.center_freq = 433.92 * 1e6
sdr.gain = gain

print("Sample rate: {}".format(sdr.sample_rate))
print("Gain: {}".format(sdr.gain))
print("Reading samples {} s".format(num_samples/2/sdr.sample_rate))


sdr.read_samples()
