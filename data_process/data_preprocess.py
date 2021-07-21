# -*- coding: utf-8 -*-

# use like
# python data_preprocess.py --clean_files C:/Users/aquam/Documents/GitHub/RnE/create_datasets/data/clean_data/TIMIT/TRAIN --mixed_files C:/Users/aquam/Documents/GitHub/RnE/create_datasets/data/mixed_data --output_ground_truth_masks D:/RnE/data/ground_truth_mask --output_mixed_stfts D:/RnE/data/mixed_stft --num_data 

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
    parser.add_argument('--num_data', type=int, required=True, default=-1)
    args = parser.parse_args()
    return args

def stft(file):
    data = librosa.load(file, sr=16000, dtype='float64')[0]
    stft = librosa.stft(data, n_fft=256, hop_length=128, in_length=256)
    return stft

def cIRM(clean_stft, mixed_stft):
    return clean_stft / mixed_stft

def save_as_csv(data, path):
    data = DataFrame(data)
    data.to_csv(path, header=False, index=False)

def data_preprocess(clean_files, mixed_files, output_ground_truth_masks, output_mixed_stfts, num_data):
    for mixed_file in os.listdir(mixed_files)[:num_data]:
        clean_file = mixed_file.split('+')[0]

        clean_stft = stft(clean_files + '/' + clean_file)
        mixed_stft = stft(mixed_files + '/' + mixed_file)
        
        save_as_csv(np.abs(data=mixed_stft), path=output_mixed_stfts + '/' + mixed_file + '-' + 'stft.csv')

        cIRM = cIRM(clean_stft, mixed_stft)
        save_as_csv(np.abs(cIRM), path=output_ground_truth_masks + '/' + mixed_file + '-' + 'ground_mask_mag.csv')

if __name__=='__main__':

    args = get_args()
    data_preprocess(args.clean_files, args.mixed_files, args.output_ground_masks, args.output_mixed_stfts, args.num_data)