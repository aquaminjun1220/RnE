import librosa
import matplotlib.pyplot as plt

mixed_path = "D:/RnE/data/estimated/TEST/SI458.WAV.wav+DKITCHEN_ch01.wav--snr0.wav"
mixed_data = librosa.load(mixed_path, sr=16000, dtype="float64")[0]

clean_path = "C:/Users/aquam/Documents/GitHub/RnE/data/clean_data/TIMIT/TEST/SI458.WAV.wav"
clean_data = librosa.load(clean_path, sr=16000, dtype="float64")[0]

plt.figure(figsize=(8,8))
plt.subplot(211)
plt.plot(mixed_data, '-b')
plt.subplot(212)
plt.plot(clean_data, '-b')
plt.show()