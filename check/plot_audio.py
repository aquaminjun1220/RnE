import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

clean_path = "C:/Users/aquam/Documents/GitHub/RnE/data/clean_data/TIMIT/TEST/SI458.WAV.wav"
mixed_path = "D:/RnE/data/mixed_data/TEST/SI458.WAV.wav+DKITCHEN_ch01.wav--snr0.wav"
estimated_path = "D:/RnE/data/estimated/TEST/SI458.WAV.wav+DKITCHEN_ch01.wav--snr0.wav"

clean_data = librosa.load(clean_path, sr=16000, dtype="float64")[0]
clean_stft = librosa.stft(clean_data, n_fft=2048, hop_length=512, win_length=2048)
clean_spec_mag = librosa.amplitude_to_db(np.abs(clean_stft))

mixed_data = librosa.load(mixed_path, sr=16000, dtype="float64")[0]
mixed_stft = librosa.stft(mixed_data, n_fft=2048, hop_length=512, win_length=2048)
mixed_spec_mag = librosa.amplitude_to_db(np.abs(mixed_stft))

estimated_data = librosa.load(estimated_path, sr=16000, dtype="float64")[0]
estimated_stft = librosa.stft(estimated_data, n_fft=2048, hop_length=512, win_length=2048)
estimated_spec_mag = librosa.amplitude_to_db(np.abs(estimated_stft))

clean_path2 = "C:/Users/aquam/Documents/GitHub/RnE/data/clean_data/TIMIT/TEST/SI458.WAV.wav"
mixed_path2 = "D:/RnE/data/mixed_data/TEST/SI458.WAV.wav+NPARK_ch01.wav--snr0.wav"
estimated_path2 = "D:/RnE/data/estimated/TEST/SI458.WAV.wav+NPARK_ch01.wav--snr0.wav"

clean_data2 = librosa.load(clean_path2, sr=16000, dtype="float64")[0]
clean_stft2 = librosa.stft(clean_data2, n_fft=2048, hop_length=512, win_length=2048)
clean_spec_mag2 = librosa.amplitude_to_db(np.abs(clean_stft2))

mixed_data2 = librosa.load(mixed_path2, sr=16000, dtype="float64")[0]
mixed_stft2 = librosa.stft(mixed_data2, n_fft=2048, hop_length=512, win_length=2048)
mixed_spec_mag2 = librosa.amplitude_to_db(np.abs(mixed_stft2))

estimated_data2 = librosa.load(estimated_path2, sr=16000, dtype="float64")[0]
estimated_stft2 = librosa.stft(estimated_data2, n_fft=2048, hop_length=512, win_length=2048)
estimated_spec_mag2 = librosa.amplitude_to_db(np.abs(estimated_stft2))

plt.figure(figsize=(15,6))
plt.rc('font', size=12)

plt.subplot(321)
librosa.display.specshow(clean_spec_mag,sr=16000,hop_length=512, cmap='plasma')
plt.ylabel("Frequency")
plt.colorbar(format='%+2.0f dB')
plt.title("Clean / Stationary")

plt.subplot(323)
librosa.display.specshow(mixed_spec_mag,sr=16000,hop_length=512, cmap='plasma')
plt.ylabel("Frequency")
plt.colorbar(format='%+2.0f dB')
plt.title("Noisy / Stationary")

plt.subplot(325)
librosa.display.specshow(estimated_spec_mag,sr=16000,hop_length=512, cmap='plasma')
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar(format='%+2.0f dB')
plt.title("Estimated / Stationary")

plt.subplot(322)
librosa.display.specshow(clean_spec_mag2,sr=16000,hop_length=512, cmap='plasma')
plt.ylabel("Frequency")
plt.colorbar(format='%+2.0f dB')
plt.title("Clean / Nonstationary")

plt.subplot(324)
librosa.display.specshow(mixed_spec_mag2,sr=16000,hop_length=512, cmap='plasma')
plt.ylabel("Frequency")
plt.colorbar(format='%+2.0f dB')
plt.title("Noisy / Nonstationary")

plt.subplot(326)
librosa.display.specshow(estimated_spec_mag2,sr=16000,hop_length=512)
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar(format='%+2.0f dB')
plt.title("Estimated / Nonstationary")

plt.show()