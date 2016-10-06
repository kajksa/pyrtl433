import sys
import time

import numpy as np
import pigpio

# GPIO pin
gpio = 17

# pigpio waves are send in microseconds
sample_rate = 250000 # Hardcoded
def samples_microsecs(samples, sample_rate):
    return int(np.round(1000000/sample_rate))
scale = samples_microsecs(1,sample_rate)

filename = sys.argv[1]

print("Loading file {}".format(filename))

widths = np.load(filename)
assert len(widths) % 2 == 0
pulses = widths[::2]
gaps = widths[1::2]

print("Pulse widths", set(pulses))
print("Gap widths", set(gaps))

pi = pigpio.pi()
pi.set_mode(gpio, pigpio.OUTPUT)
pi.wave_clear()
waves = {}
for p,g in zip(pulses, gaps):
    if (p,g) not in waves:
        pi.wave_add_generic([pigpio.pulse(1<<gpio, 0, scale*p), pigpio.pulse(0, 1<<gpio, scale*g)]) # TODO: sample rate!
        waves[(p,g)]  = pi.wave_create()
print("Number of waves: ", len(waves))

# Create wave chain
chain = []
for p,g in zip(pulses, gaps):
    chain.append(waves[(p,g)])
    if len(chain)>575: # TODO: Something with a hardware limit around ~600
        print("Warning: wave chain too long")
        break

# Send waves
print("Sending...")
pi.wave_chain(chain)
while pi.wave_tx_busy():
    time.sleep(0.1)
pi.stop()
print("DONE")
