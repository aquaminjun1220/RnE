from pypesq import pesq
from scipy.io import wavfile
import numpy as np
import wave

def cal_amp(wf):
    buffer = wf.readframes(wf.getnframes())
    # The dtype depends on the value of pulse-code modulation. The int16 is set for 16-bit PCM.
    amptitude = (np.frombuffer(buffer, dtype="int16")).astype(np.float64)
    return amptitude

clean_file = "C:/Users/aquam/Documents/GitHub/RnE/data/clean_data/TIMIT/TEST/SA1.WAV.wav"
est_file = "D:\RnE\data\estimated\TEST_stat/128_128_basic\SA1.WAV.wav+static_0_10.wav--snr10.0.wav"

clean_wav = wave.open(clean_file, "r")
est_wav = wave.open(est_file, "r")
clean = cal_amp(clean_wav)
est = cal_amp(est_wav)
print("hello")

"""rate, clean = wavfile.read(clean_file)
rate, est = wavfile.read(est_file)"""

print("zz")
print(pesq(clean, est, fs=16000, normalize=False))
print("yabal")