# -*- coding: utf-8 -*-

# use like
# python create_data.py --clean_files C:/Users/aquamj/Documents/GitHub/RnE/create_datasets/data/clean_data/TIMIT/TRAIN --mixed_files C:/Users/aquamj/Documents/GitHub/RnE/create_datasets/data/mixed_data --output_ground_truth_masks D:/RnE/data/ground_truth_mask --output_mixed_stfts D:/RnE/data/mixed_stft

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
    parser.add_argument('--output_ground_truth_masks', type=str, required=True)
    parser.add_argument('--output_mixed_stfts', type=str, required=True)
    args = parser.parse_args()
    return args


if __name__=='__main__':

    args = get_args()

    for mixed_file in os.listdir(args.mixed_files)[7500:]:
        clean_file = mixed_file.split('+')[0]

        clean_data = librosa.load(args.clean_files + '/' + clean_file, sr=16000, dtype="float64")[0]
        mixed_data = librosa.load(args.mixed_files + '/' + mixed_file, sr=16000, dtype="float64")[0]


        clean_stft = librosa.stft(clean_data, n_fft=256, hop_length=128, win_length=256)
        mixed_stft = librosa.stft(mixed_data, n_fft=256, hop_length=128, win_length=256)
        
        mixed_stft_mag = np.abs(mixed_stft)

        mixed_stft_mag = DataFrame(mixed_stft_mag)

        mixed_stft_mag.to_csv(args.output_mixed_stfts + '/' + mixed_file + '-' + 'stft.csv', header=False, index=False)

        #cIRM = clean_stft / mixed_stft

        #cIRM_mag = np.abs(cIRM)

        #cIRM_mag = DataFrame(cIRM_mag)

        #cIRM_mag.to_csv(args.output_ground_truth_masks + '/' + mixed_file + '-' + 'ground_mask_mag.csv', header=False, index=False)
