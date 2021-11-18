import numpy as np
import librosa
import wave
import os

class opt_Model():
    def __init__(self):
        pass
    
    def estimate2(self, clean_file_paths, mixed_file_paths, mixed_file_path, output_paths, error=0):

        clean_file_path = mixed_file_path.split("+")[0]
        clean_file = wave.open(clean_file_paths+'/'+clean_file_path, mode="rb")
        est_file_path = output_paths+'/'+mixed_file_path
        est_file = wave.open(est_file_path, mode="wb")

        clean_stft = librosa.stft(librosa.load(clean_file_paths+'/'+clean_file_path, sr=16000, dtype="float64")[0], n_fft=256, hop_length=128)

        cIRM = np.ones(shape=np.shape(clean_stft))
        if error!=0:
            cIRM = np.random.normal(loc=1, scale=error, size=np.shape(clean_stft))
        est_stft = clean_stft * cIRM
        est_data = librosa.istft(est_stft, hop_length=128, win_length=256)
        est_data = est_data*2**15

        est_file.setparams(clean_file.getparams())
        est_file.writeframes(est_data.astype(np.int16).tobytes())
        est_file.close()
    
    def estimate(self, clean_data, mixed_data, error=0):

        clean_stft = librosa.stft(clean_data, n_fft=256, hop_length=128)

        cIRM = np.ones(shape=np.shape(clean_stft))
        if error!=0:
            cIRM = np.random.normal(loc=1, scale=error, size=np.shape(clean_stft))
        est_stft = clean_stft * cIRM
        est_data = librosa.istft(est_stft, hop_length=128, win_length=256)
        return est_data

    def run(self, clean_file_paths, mixed_file_paths, output_paths, num_data=-1, error=0):
        mixed_ind = 0
        while (num_data == -1 and mixed_ind <= len(os.listdir(mixed_file_paths)-1)) or mixed_ind <= num_data-1:
            mixed_file_path = os.listdir(mixed_file_paths)[mixed_ind]
            clean_file_path = mixed_file_path.split("+")[0]
            est_file_path = output_paths + mixed_file_path

            mixed_data = librosa.load(mixed_file_paths+mixed_file_path, sr=16000, dtype='float64')[0]
            clean_data = librosa.load(clean_file_paths+clean_file_path, sr=16000, dtype='float64')[0]
            est_data = self.estimate(clean_data=clean_data, mixed_data=mixed_data, error=error)
            est_data *= 2**15

            with wave.open(clean_file_paths+clean_file_path, mode="r") as clean_file:
                est_file = wave.open(est_file_path, mode="w")
                est_file.setparams(clean_file.getparams())
                est_file.writeframes(est_data.astype(np.int16).tobytes())
                est_file.close()

    def run2(self, clean_file_paths, mixed_file_paths, num_data=-1, error=0):
        mixed_ind = 0
        while (num_data == -1 and mixed_ind <= len(os.listdir(mixed_file_paths)-1)) or mixed_ind <= num_data-1:
            mixed_file_path = os.listdir(mixed_file_paths)[mixed_ind]
            clean_file_path = mixed_file_path.split("+")[0]
            snr = int(mixed_file_path[mixed_file_path.index("snr")+3 : -3])

            mixed_data = librosa.load(mixed_file_paths+mixed_file_path, sr=16000, dtype='float64')[0]
            clean_data = librosa.load(clean_file_paths+clean_file_path, sr=16000, dtype='float64')[0]
            est_data = self.estimate(clean_data=clean_data, mixed_data=mixed_data, error=error)





class opt_no_phase_Model(opt_Model):
    def estimate(self, clean_data, mixed_data, error=0):

        clean_stft = librosa.stft(clean_data, n_fft=256, hop_length=128)
        mixed_stft = librosa.stft(mixed_data, n_fft=256, hop_length=128)

        cIRM = clean_stft/mixed_stft
        if error!=0:
            cIRM = cIRM * np.random.normal(loc=1, scale=error, size=np.shape(cIRM))
        cIRM_mag = np.abs(cIRM)
        est_stft = mixed_stft * cIRM_mag
        est_data = librosa.istft(est_stft, hop_length=128, win_length=256)

        return est_data