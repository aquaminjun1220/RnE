# -*- coding: utf-8 -*-

# use like
# python create_mixed_audio_file.py --clean_files C:/Users/aquam/Documents/GitHub/RnE/create_datasets/data/clean_data/TIMIT/TRAIN/ --noise_files C:/Users/aquam/Documents/GitHub/RnE/create_datasets/data/noise_data/DEMAND/ --output_mixed_files D:/RnE/data/mixed_data/TRAIN/ --snrs -10 0 10

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

def create_mixed_audio_file(clean_files, noise_files, output_mixed_files, snrs, num_clean, num_noise):
	clean_file_list = os.listdir(clean_files)
	noise_file_list = os.listdir(noise_files)
	clean_ind = 0
	while (num_clean == -1 and clean_ind <= len(clean_file_list)-1) or clean_ind <= num_clean-1:
		clean_file = clean_file_list[clean_ind]
		if not clean_file.endswith(".wav"):
			continue
		noise_ind = 0
		while (num_noise == -1 and noise_ind <= len(noise_file_list)-1) or noise_ind <= num_noise-1:
			noise_file = noise_file_list[noise_ind]
			if not noise_file.endswith(".wav"):
				continue
			clean_wav = wave.open(clean_files + clean_file, "r")
			noise_wav = wave.open(noise_files + noise_file, "r")

			clean_amp = cal_amp(clean_wav)
			noise_amp = cal_amp(noise_wav)

			clean_rms = cal_rms(clean_amp)

			start = random.randint(0, len(noise_amp)-len(clean_amp))
			divided_noise_amp = noise_amp[start: start + len(clean_amp)]
			noise_rms = cal_rms(divided_noise_amp)
			
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
				output_mixed_file = output_mixed_files + clean_file + '+' + noise_file + '--snr' + str(snr) + '.wav'
				save_waveform(output_mixed_file, clean_wav.getparams(), mixed_amp)
			noise_ind += 1
		print("c:", clean_ind)	
		clean_ind += 1

if __name__ == '__main__':
	args = get_args()

	clean_files = args.clean_files
	noise_files = args.noise_files
	output_mixed_files = args.output_mixed_files
	snrs = args.snrs

	create_mixed_audio_file(clean_files, noise_files, output_mixed_files, snrs)
