import numpy as np
import librosa
import wave
import os
import pandas as pd
import array

class opt_Model():
    def __init__(self, clean_file_paths="./data\clean_data\TIMIT\TRAIN", mixed_file_paths="D:\RnE\data\mixed_data\TRAIN"):
        self.clean_file_paths = clean_file_paths
        self.mixed_file_paths = mixed_file_paths
    
    def estimate(self, mixed_file_path, output_paths):
        clean_file_paths = self.clean_file_paths
        mixed_file_paths = self.mixed_file_paths

        clean_file_path = mixed_file_path.split("+")[0]
        clean_file = wave.open(clean_file_paths+'/'+clean_file_path, mode="rb")
        est_file_path = output_paths+'/'+clean_file_path
        est_file = wave.open(est_file_path, mode="wb")

        est_file.setparams(clean_file.getparams())
        est_file.writeframes(clean_file.readframes(clean_file.getnframes()))
        est_file.close()

    def run(self, output_paths, num_data=-1):
        clean_file_paths = self.clean_file_paths
        mixed_file_paths = self.mixed_file_paths

        if num_data!=-1:
            for mixed_file_path in os.listdir(mixed_file_paths)[:num_data]:
                self.estimate(mixed_file_path, output_paths)
        else:
            for mixed_file_path in os.listdir(mixed_file_paths):
                self.estimate(mixed_file_path, output_paths)

class opt_no_phase_Model(opt_Model):
    def estimate(self, mixed_file_path, output_paths):
        clean_file_paths = self.clean_file_paths
        mixed_file_paths = self.mixed_file_paths

        clean_file_path = mixed_file_path.split("+")[0]
        clean_file = wave.open(clean_file_paths+'/'+clean_file_path, mode="rb")
        est_file_path = output_paths+'/'+mixed_file_path
        est_file = wave.open(est_file_path, mode="wb")

        clean_stft = librosa.stft(librosa.load(clean_file_paths+'/'+clean_file_path, sr=16000, dtype="float64")[0], n_fft=256, hop_length=128)
        mixed_stft = librosa.stft(librosa.load(mixed_file_paths+'/'+mixed_file_path, sr=16000, dtype="float64")[0], n_fft=256, hop_length=128)

        cIRM = clean_stft/mixed_stft
        cIRM_mag = np.abs(cIRM)
        est_stft_mag = cIRM_mag * np.abs(mixed_stft)
        est_stft_phase = np.angle(mixed_stft)
        est_stft = est_stft_mag * np.exp(1j*est_stft_phase)
        est_data = librosa.istft(est_stft, hop_length=128, win_length=256)
        est_data = est_data*2**15

        est_file.setparams(clean_file.getparams())
        est_file.writeframes(est_data.astype(np.int16).tobytes())
        est_file.close()

