# python STFT.py --clean_file C:/Users/aquamj/Documents/GitHub/RnE/create_datasets/data/clean_data/TIMIT/TRAIN/DR1/FCJF0/SA1.WAV.wav --noise_file C:/Users/aquamj/Documents/GitHub/RnE/create_datasets/data/noise_data/DEMAND/DKITCHEN/ch01.wav --output_mixed_file C:/Users/aquamj/Documents/GitHub/RnE/create_datasets/data/output_mixed_data/SA1.WAV.wav+ch01.wav--snr10.wav --output_re_file C:/Users/aquamj/Documents/GitHub/RnE/create_datasets/data/output_mixed_data/AA.wav 

import librosa
import numpy as np
import argparse
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_files', type=str, required=True)
    parser.add_argument('--mixed_files', type=str, required=True)
    args = parser.parse_args()
    return args


if __name__=='__main__':

    args = get_args()

    for mixed_file in os.listdir(args.mixed_files):
        clean_file = mixed_data.split('+')[0]

        clean_data = librosa.load(args.clean_files + '/' + clean_file, sr=16000, dtype="float64")[0]
        mixed_data = librosa.load(args.mixed_files + '/' + mixed_file, sr=16000, dtype="float64")[0]


        clean_stft = librosa.stft(clean_data, n_fft=128, hop_length=64, win_length=128)
        mixed_stft = librosa.stft(mixed_data, n_fft=128, hop_length=64, win_length=128)

        cIRM = clean_stft / mixed_stft

        cIRM_real = np.real(cIRM)
        cIRM_imag = np.imag(cIRM)
    


