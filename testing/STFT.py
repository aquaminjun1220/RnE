# python STFT.py --clean_file C:/Users/aquamj/Documents/GitHub/RnE/create_datasets/data/clean_data/TIMIT/TRAIN/DR1/FCJF0/SA1.WAV.wav --noise_file C:/Users/aquamj/Documents/GitHub/RnE/create_datasets/data/noise_data/DEMAND/DKITCHEN/ch01.wav --output_mixed_file C:/Users/aquamj/Documents/GitHub/RnE/create_datasets/data/output_mixed_data/SA1.WAV.wav+ch01.wav--snr10.wav --output_re_file C:/Users/aquamj/Documents/GitHub/RnE/create_datasets/data/output_mixed_data/AA.wav 

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import argparse
import wave
import array

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_file', type=str, required=True)
    parser.add_argument('--noise_file', type=str, required=True)
    parser.add_argument('--output_mixed_file', type=str, required=True)
    parser.add_argument('--output_re_file', type=str, required=True)
    parser.add_argument('--plot', type=bool, default=False, required=False)
    args = parser.parse_args()
    return args


if __name__=='__main__':

    args = get_args()
    
    clean_data = librosa.load(args.clean_file, sr=16000, dtype="float64")[0]
    wf = wave.open(args.clean_file, 'r')
    noise_data = librosa.load(args.noise_file, sr=16000, dtype="float64")[0]
    mixed_data = librosa.load(args.output_mixed_file, sr=16000, dtype="float64")[0]

    print("clean_data's datatype is {type}".format(type=type(clean_data[0])))
    print("clean_data's shape is {shape}".format(shape=np.shape(clean_data)))

    clean_stft = librosa.stft(clean_data, n_fft=2048, hop_length=512, win_length=2048)
    noise_stft = librosa.stft(noise_data, n_fft=2048, hop_length=512, win_length=2048)
    mixed_stft = librosa.stft(mixed_data, n_fft=2048, hop_length=512, win_length=2048)

    print("clean_stft's datatype is {type}".format(type=type(clean_stft[0][0])))
    print("clean_stft's shape is {shape}".format(shape=np.shape(clean_stft)))

    cIRM = clean_stft / mixed_stft
    re_stft = mixed_stft * cIRM

    clean_spec_r = librosa.amplitude_to_db(np.real(clean_stft))
    clean_spec_i = librosa.amplitude_to_db(np.imag(clean_stft))
    noise_spec_r = librosa.amplitude_to_db(np.real(noise_stft))
    noise_spec_i = librosa.amplitude_to_db(np.imag(noise_stft))
    mixed_spec_r = librosa.amplitude_to_db(np.real(mixed_stft))
    mixed_spec_i = librosa.amplitude_to_db(np.imag(mixed_stft))
    re_spec_r = librosa.amplitude_to_db(np.real(re_stft))
    re_spec_i = librosa.amplitude_to_db(np.imag(re_stft))

    cIRM_spec_r = librosa.amplitude_to_db(np.real(cIRM))
    cIRM_spec_i = librosa.amplitude_to_db(np.imag(cIRM))

    re_data = librosa.istft(re_stft, hop_length=512, win_length=2048)
    re_data = re_data*(2**15)

    output_file = wave.Wave_write(args.output_re_file)
    output_file.setparams(wave.open(args.output_mixed_file, 'r').getparams())
    output_file.writeframes(re_data.astype(np.int16).tobytes())
    output_file.close()

    if args.plot:
        plt.figure(figsize=(16,10))
        plt.subplot(421)
        librosa.display.specshow(clean_spec_r,sr=16000,hop_length=512)
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.colorbar(format='%+2.0f dB')
        plt.title("Clean Data Spectrogram (Real)")
        plt.subplot(422)
        librosa.display.specshow(clean_spec_i,sr=16000,hop_length=512)
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.colorbar(format='%+2.0f dB')
        plt.title("Clean Data Spectrogram (Imag)")
        plt.subplot(423)
        librosa.display.specshow(cIRM_spec_r,sr=16000,hop_length=512)
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.colorbar(format='%+2.0f dB')
        plt.title("cIRM Spectrogram (Real)")
        plt.subplot(424)
        librosa.display.specshow(cIRM_spec_i,sr=16000,hop_length=512)
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.colorbar(format='%+2.0f dB')
        plt.title("cIRM Spectrogram (Imag)")
        plt.subplot(425)
        librosa.display.specshow(mixed_spec_r,sr=16000,hop_length=512)
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.colorbar(format='%+2.0f dB')
        plt.title("Mixed Data Spectrogram (Real)")
        plt.subplot(426)
        librosa.display.specshow(mixed_spec_i,sr=16000,hop_length=512)
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.colorbar(format='%+2.0f dB')
        plt.title("Mixed Data Spectrogram (Imag)")
        plt.subplot(427)
        librosa.display.specshow(re_spec_r,sr=16000,hop_length=512)
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.colorbar(format='%+2.0f dB')
        plt.title("Re Data Spectrogram (Real)")
        plt.subplot(428)
        librosa.display.specshow(re_spec_i,sr=16000,hop_length=512)
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.colorbar(format='%+2.0f dB')
        plt.title("Re Data Spectrogram (Imag)")
        plt.show()

