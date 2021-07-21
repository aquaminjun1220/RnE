import librosa
import wave
import array
import numpy as np

input_path = 'D:/RnE/data/mixed_data/TEST/SA1.WAV.wav+DLIVING_ch05.wav--snr-10.wav'
output_path='D:/RnE/data/mixed/TEST/SA1.WAV.wav+DLIVING_ch05.wav--snr-10.wav'
data_lib = librosa.load(input_path, sr=16000, dtype='float64')[0]
data_wav = wave.open(input_path, 'r')
data_lib = data_lib*2**15

output_file = wave.Wave_write(output_path)
output_file.setparams(data_wav.getparams()) #nchannels, sampwidth, framerate, nframes, comptype, compname
output_file.writeframes(array.array('h', data_lib.astype(np.int16)).tobytes() )
output_file.close()

print(array.array('h', data_lib.astype(np.int16)))