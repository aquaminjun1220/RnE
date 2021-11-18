from scipy.io import wavfile
from pesq import pesq
import librosa
import time
from measure_SNR_LSD import comp_SNR, comp_LSD3


rate, ref = wavfile.read("./data\clean_data\TIMIT\TRAIN\SA1.WAV.wav")
rate, deg = wavfile.read("D:/RnE/data/mixed_data/for_final/TRAIN/SA1.WAV.wav+DKITCHEN_ch01.wav--snr15.wav")
clean = ref/(2**15)
est = deg/(2**15)

time1 = time.time()
for i in range(10):
    pesq(rate, ref, deg, 'wb')
time2 = time.time()
print(time2-time1)

time1 = time.time()
for i in range(10):
    pesq(rate, ref, deg, 'nb')
time2 = time.time()
print(time2-time1)

time1 = time.time()
for i in range(10):
    comp_SNR(est, clean)
    comp_LSD3(clean, est)
time2 = time.time()
print(time2-time1)