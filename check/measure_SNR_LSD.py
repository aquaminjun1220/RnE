#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 5/30/2019 3:05 PM
# @Author  : HOU NANA
# @Site    : http://github.com/nanahou
# @File    : measure_SNR_LSD.py
"""import np"""
import numpy as np
import librosa
import os
"""from scripts.extract_LPS_CMVN import get_power_spec"""

def comp_SNR(x, y):
    """
       Compute SNR (signal to noise ratio)
       Arguments:
           x: vector, enhanced signal
           y: vector, reference signal(ground truth)
    """
    ref = np.power(y, 2)
    if len(x) == len(y):
        diff = np.power(x-y, 2)
    else:
        stop = min(len(x), len(y))
        diff = np.power(x[:stop] - y[:stop], 2)

    ratio = np.sum(ref) / np.sum(diff)
    value = 10*np.log10(ratio)

    return value

def comp_SNR2(clean_file_path, mixed_file_path):
    clean_data = librosa.load(clean_file_path, None, mono=True, offset=0.0, dtype=np.float32)[0]
    mixed_data = librosa.load(mixed_file_path, None, mono=True, offset=0.0, dtype=np.float32)[0]
    return comp_SNR(x=mixed_data, y=clean_data)

def comp_LSD(x, y):
    """
       Compute LSD (log spectral distance)
       Arguments:
           x: vector (np.Tensor), enhanced signal
           y: vector (np.Tensor), reference signal(ground truth)
    """
    if len(x) == len(y):
        diff = np.power(x-y, 2)
    else:
        stop = min(len(x), len(y))
        diff = np.power(x[:stop] - y[:stop], 2)

    sum_freq = np.sqrt(np.sum(diff, axis=1) / diff.shape[1])
    value = np.sum(sum_freq, axis=0) / sum_freq.shape[0]

    return value

def comp_LSD2(clean_file_path, mixed_file_path):
    clean_data = librosa.load(clean_file_path, None, mono=True, offset=0.0, dtype=np.float32)[0]
    mixed_data = librosa.load(mixed_file_path, None, mono=True, offset=0.0, dtype=np.float32)[0]

    # compute LSD
    fft_len_16k, frame_shift_16k = 512, 256

    # extract magnitude and power
    power_16k = np.abs(librosa.stft(clean_data, n_fft=fft_len_16k, hop_length=frame_shift_16k))
    power_16k = power_16k.astype(np.float)

    power_ext = np.abs(librosa.stft(mixed_data, n_fft=fft_len_16k, hop_length=frame_shift_16k))
    power_ext = power_ext.astype(np.float)

    log_16k = np.log(power_16k)
    log_ext = np.log(power_ext)

    return comp_LSD(log_ext, log_16k)

def comp_LSD3(clean_data, mixed_data):
    """as librosa sequence NOT WAVE"""
    # compute LSD
    fft_len_16k, frame_shift_16k = 512, 256

    # extract magnitude and power
    power_16k = np.abs(librosa.stft(clean_data, n_fft=fft_len_16k, hop_length=frame_shift_16k))
    power_16k = power_16k.astype(np.float)

    power_ext = np.abs(librosa.stft(mixed_data, n_fft=fft_len_16k, hop_length=frame_shift_16k))
    power_ext = power_ext.astype(np.float)

    log_16k = np.log(power_16k)
    log_ext = np.log(power_ext)

    return comp_LSD(log_ext, log_16k)

def main():

    wav_16k = 'C:/Users/aquam\Documents\GitHub\RnE\data\clean_data\TIMIT\TRAIN/'
    # extend_16k = '/data/disk3/hounana/Valentini-Botinhao_16k/formatted_28spk/noisy8k_clean16k/raw_wav/clean_testset_wav_re16k/'
    # wav_16k = '/data/disk3/hounana/Valentini-Botinhao_16k/formatted_28spk/noisy_re16k_clean16k/tt/clean/'
    extend_16k = 'D:\RnE\data\opt\opt_no_phase\TRAIN/0.5/'
    extend_list = [x for x in os.listdir(extend_16k) if x.endswith(".wav")]

    sum_snr_enhan = 0.0
    sum_lsd_enhan = 0.0

    for item in extend_list:
        item_org16k = wav_16k + item[:11]
        item_extend = extend_16k + item
        # item_extend = extend_16k + item[:-4] + '.wav..pr.wav'

        # compute SNR
        org_sig, org_sr = librosa.load(item_org16k, None, mono=True, offset=0.0, dtype=np.float32)
        ext_sig, ext_sr = librosa.load(item_extend, None, mono=True, offset=0.0, dtype=np.float32)
        x = ext_sig
        y = org_sig
        value_snr = comp_SNR(x, y)
        sum_snr_enhan += value_snr

        # compute LSD
        fft_len_16k, frame_shift_16k = 512, 256

        # extract magnitude and power
        power_16k = np.abs(librosa.stft(org_sig, n_fft=fft_len_16k, hop_length=frame_shift_16k))
        """power_16k = get_power_spec(item_org16k, fft_len_16k, frame_shift_16k)"""
        power_16k = power_16k.astype(np.float)

        power_ext = np.abs(librosa.stft(ext_sig, n_fft=fft_len_16k, hop_length=frame_shift_16k))
        """power_ext = get_power_spec(item_extend, fft_len_16k, frame_shift_16k)"""
        power_ext = power_ext.astype(np.float)

        log_16k = np.log(power_16k)
        log_ext = np.log(power_ext)

        value_lsd = comp_LSD(log_ext, log_16k)

        sum_lsd_enhan += value_lsd


    avg_snr_enhan = sum_snr_enhan / len(extend_list)
    avg_lsd_enhan = sum_lsd_enhan / len(extend_list)

    print('avg_snr_enhan %f, avg_lsd_enhan %f' % (avg_snr_enhan, avg_lsd_enhan))




if __name__ == '__main__':
    main()