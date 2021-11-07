import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_cIRM(path):
    cIRM = np.array(pd.read_csv(path, header=None))
    cIRM = librosa.amplitude_to_db(cIRM)
    plt.figure(figsize=(6,8))
    librosa.display.specshow(cIRM, sr=16000, hop_length=128, cmap="plasma")
    plt.show()

plot_cIRM("D:\RnE\data\ground_truth_mask\TEST_stat\SA1.WAV.wav+static_0_1.wav--snr10.0.wav-ground_mask_mag.csv")