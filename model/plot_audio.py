import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

clean_path = "C:/Users/aquam/Documents/GitHub/RnE/data/clean_data/TIMIT/TEST/SI458.WAV.wav"
mixed_path = "D:/RnE/data/mixed_data/TEST/SI458.WAV.wav+DLIVING_ch01.wav--snr0.wav"
estimated_path = "D:/RnE/data/estimated/TEST/SI458.WAV.wav+DLIVING_ch01.wav--snr0.wav"

clean_data = librosa.load(clean_path, sr=16000, dtype="float64")[0]
clean_stft = librosa.stft(clean_data, n_fft=2048, hop_length=512, win_length=2048)
clean_spec_mag = librosa.amplitude_to_db(np.abs(clean_stft))

mixed_data = librosa.load(mixed_path, sr=16000, dtype="float64")[0]
mixed_stft = librosa.stft(mixed_data, n_fft=2048, hop_length=512, win_length=2048)
mixed_spec_mag = librosa.amplitude_to_db(np.abs(mixed_stft))

estimated_data = librosa.load(estimated_path, sr=16000, dtype="float64")[0]
estimated_stft = librosa.stft(estimated_data, n_fft=2048, hop_length=512, win_length=2048)
estimated_spec_mag = librosa.amplitude_to_db(np.abs(estimated_stft))

plt.figure(figsize=(8,6))

plt.subplot(311)
librosa.display.specshow(clean_spec_mag,sr=16000,hop_length=512)
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar(format='%+2.0f dB')
plt.title("Clean Data Spectrogram (Mag)")

plt.subplot(312)
librosa.display.specshow(mixed_spec_mag,sr=16000,hop_length=512)
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar(format='%+2.0f dB')
plt.title("Mixed Data Spectrogram (Mag)")

plt.subplot(313)
librosa.display.specshow(estimated_spec_mag,sr=16000,hop_length=512)
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar(format='%+2.0f dB')
plt.title("Estimated Data Spectrogram (Mag)")

plt.show()