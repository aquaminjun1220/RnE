from data_process.data_preprocess import data_preprocess
from data_process.create_mixed_audio_file import create_mixed_audio_file
import argparse

"""python prep.py --clean_files ./data/clean_data/TIMIT/TRAIN --noise_files ./data/noise_data/stat_noise --output_mixed_files D:/RnE/data/mixed_data/TRAIN_stat --snrs -10 0 10 --output_ground_truth_masks D:/RnE/data/ground_truth_mask/TRAIN_stat --output_mixed_stfts D:/RnE/data/mixed_stft/TRAIN_stat --num_data -1
"""

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_files', type=str, required=True)
    parser.add_argument('--noise_files', type=str, required=True)
    parser.add_argument('--output_mixed_files', type=str, default='', required=True)
    parser.add_argument('--snrs', type=float, nargs='+', default=[-10, -5, 0, 5, 10], required=False)
    parser.add_argument('--output_ground_truth_masks', type=str, required=True)
    parser.add_argument('--output_mixed_stfts', type=str, required=True)
    parser.add_argument('--num_clean', type=int, required=True, default=-1)
    parser.add_argument('--num_noise', type=int, required=True, default=-1)
    args = parser.parse_args()
    return args

def prep(clean_files, noise_files,  output_mixed_files, snrs, output_ground_truth_masks, output_mixed_stfts, num_clean, num_noise):
    create_mixed_audio_file(clean_files=clean_files, noise_files=noise_files, output_mixed_files=output_mixed_files, snrs=snrs, num_clean=num_clean, num_noise=num_noise)
    data_preprocess(clean_files=clean_files, mixed_files=output_mixed_files, output_ground_truth_masks=output_ground_truth_masks, output_mixed_stfts=output_mixed_stfts, num_data=-1)

if __name__=='__main__':
    args = get_args()
    clean_files = args.clean_files
    noise_files = args.noise_files
    snrs = args.snrs
    output_mixed_files = args.output_mixed_files
    output_ground_truth_masks = args.output_ground_truth_masks
    output_mixed_stfts = args.output_mixed_stfts
    num_data = args.num_data
    prep(clean_files=clean_files, noise_files=noise_files, output_mixed_files=output_mixed_files, snrs=snrs, output_ground_truth_masks=output_ground_truth_masks, output_mixed_stfts=output_mixed_stfts, num_data=num_data)