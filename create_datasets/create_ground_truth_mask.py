# -*- coding: utf-8 -*-

# use like
# python create_ground_truth_mask.py --clean_files C:/Users/aquamj/Documents/GitHub/RnE/create_datasets/data/clean_data/TIMIT/TRAIN --mixed_files C:/Users/aquamj/Documents/GitHub/RnE/create_datasets/data/mixed_data --output_ground_truth_mask C:/Users/aquamj/Documents/GitHub/RnE/create_datasets/data/ground_truth_mask

import librosa
import numpy as np
import argparse
import os
import pandas as pd
from pandas import DataFrame

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_files', type=str, required=True)
    parser.add_argument('--mixed_files', type=str, required=True)
    parser.add_argument('--output_ground_truth_mask', type=str, required=True)
    args = parser.parse_args()
    return args


if __name__=='__main__':

    args = get_args()

    for mixed_file in os.listdir(args.mixed_files):
        clean_file = mixed_file.split('+')[0]

        clean_data = librosa.load(args.clean_files + '/' + clean_file, sr=16000, dtype="float64")[0]
        mixed_data = librosa.load(args.mixed_files + '/' + mixed_file, sr=16000, dtype="float64")[0]


        clean_stft = librosa.stft(clean_data, n_fft=128, hop_length=64, win_length=128)
        mixed_stft = librosa.stft(mixed_data, n_fft=128, hop_length=64, win_length=128)

        cIRM = clean_stft / mixed_stft

        cIRM_real = np.real(cIRM)
        cIRM_imag = np.imag(cIRM)
    
        cIRM_real = DataFrame(cIRM_real)
        cIRM_imag = DataFrame(cIRM_imag)

        cIRM_real.to_csv(args.output_ground_truth_mask + '/' + mixed_file + '-' + 'ground_mask_real.csv', header=False, index=False)
        cIRM_imag.to_csv(args.output_ground_truth_mask + '/' + mixed_file + '-' + 'ground_mask_imag.csv', header=False, index=False)
