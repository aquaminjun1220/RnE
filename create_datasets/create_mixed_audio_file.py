# -*- coding: utf-8 -*-

# use like
# python create_mixed_audio_file.py --clean_files C:/Users/aquamj/Documents/GitHub/RnE/create_datasets/data/clean_data/TIMIT/TRAIN --noise_files C:/Users/aquamj/Documents/GitHub/RnE/create_datasets/data/noise_data/DEMAND --output_mixed_files C:/Users/aquamj/Documents/GitHub/RnE/create_datasets/data/mixed_data --snrs -10 0 10

import argparse
import array
import math
import numpy as np
import random
import wave
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_files', type=str, required=True)
    parser.add_argument('--noise_files', type=str, required=True)
    parser.add_argument('--output_mixed_files', type=str, default='', required=True)
    parser.add_argument('--snrs', type=float, nargs='+', default=[-10, -5, 0, 5, 10], required=False)
    args = parser.parse_args()
    return args

def cal_adjusted_rms(clean_rms, snr):
    a = float(snr) / 20
    noise_rms = clean_rms / (10**a) 
    return noise_rms

def cal_amp(wf):
    buffer = wf.readframes(wf.getnframes())
    # The dtype depends on the value of pulse-code modulation. The int16 is set for 16-bit PCM.
    amptitude = (np.frombuffer(buffer, dtype="int16")).astype(np.float64)
    return amptitude

def cal_rms(amp):
    return np.sqrt(np.mean(np.square(amp), axis=-1))

def save_waveform(output_path, params, amp):
    output_file = wave.Wave_write(output_path)
    output_file.setparams(params) #nchannels, sampwidth, framerate, nframes, comptype, compname
    output_file.writeframes(array.array('h', amp.astype(np.int16)).tobytes() )
    output_file.close()

if __name__ == '__main__':
    args = get_args()

    clean_files = args.clean_files
    noise_files = args.noise_files

    clean_files = os.listdir(clean_files)[:500]
    noise_files = os.listdir(noise_files)[:10]

    for noise_file in noise_files:
        if not noise_file.endswith(".wav"):
            continue
        for clean_file in clean_files:
            if not clean_file.endswith(".wav"):
                continue
            clean_wav = wave.open(args.clean_files + '/' + clean_file, "r")
            noise_wav = wave.open(args.noise_files + '/' + noise_file, "r")

            clean_amp = cal_amp(clean_wav)
            noise_amp = cal_amp(noise_wav)

            clean_rms = cal_rms(clean_amp)

            start = random.randint(0, len(noise_amp)-len(clean_amp))
            divided_noise_amp = noise_amp[start: start + len(clean_amp)]
            noise_rms = cal_rms(divided_noise_amp)

            snrs = [-10, 0, 10]
            for snr in snrs:
                adjusted_noise_rms = cal_adjusted_rms(clean_rms, snr)
                
                adjusted_noise_amp = divided_noise_amp * (adjusted_noise_rms / noise_rms) 
                mixed_amp = (clean_amp + adjusted_noise_amp)

                #Avoid clipping noise
                max_int16 = np.iinfo(np.int16).max
                min_int16 = np.iinfo(np.int16).min
                if mixed_amp.max(axis=0) > max_int16 or mixed_amp.min(axis=0) < min_int16:
                    if mixed_amp.max(axis=0) >= abs(mixed_amp.min(axis=0)): 
                        reduction_rate = max_int16 / mixed_amp.max(axis=0)
                    else :
                        reduction_rate = min_int16 / mixed_amp.min(axis=0)
                    mixed_amp = mixed_amp * (reduction_rate)
                    clean_amp = clean_amp * (reduction_rate)

                ## make a new name for output_mixed_file
                output_mixed_file = args.output_mixed_files + '/' + clean_file + '+' + noise_file + '--snr' + str(snr) + '.wav'
                save_waveform(output_mixed_file, clean_wav.getparams(), mixed_amp)
