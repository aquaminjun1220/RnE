import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import os
import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import librosa
import wave
import array
# from create_datasets.create_data import create_data

class Dataset():
    def __init__(self, mixed_stfts, ground_truth_masks, shuffle=False, batch_size=1, init=False, flat=True, clean_files=None, mixed_files=None):
        self.mixed_stfts = mixed_stfts
        self.ground_truth_masks = ground_truth_masks
        self.flat = flat
        #if init:
            #create_data(clean_files=clean_files, mixed_files=mixed_files, output_ground_truth_masks=ground_truth_masks, output_mixed_stfts=mixed_stfts)
        self.dataset = tf.data.Dataset.from_generator(self._gen, output_signature=(tf.TensorSpec(shape=(None, 129), dtype=np.float32), tf.TensorSpec(shape=(None, 129), dtype=np.float32)))
        if shuffle:
            self.dataset = self.dataset.shuffle(20000, reshuffle_each_iteration=True)
        self.dataset = self.dataset.padded_batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    def _gen(self):
        mixed_stfts = self.mixed_stfts
        ground_truth_masks = self.ground_truth_masks
        flat = self.flat
        stfts = os.listdir(mixed_stfts)
        masks = os.listdir(ground_truth_masks)
        if flat:
            i = 0
            while i<len(stfts):
                stft = stfts[i]
                mask = masks[i]
                
                stft = pd.read_csv(mixed_stfts+'/'+stft, header=None)
                mask = pd.read_csv(ground_truth_masks+'/'+mask, header=None)
                
                yield (np.transpose(stft), np.transpose(mask))

                i+=1
        else:
            i = 0
            while i<len(stfts):
                stft = stfts[i]
                mask = masks[i]
                
                stft = pd.read_csv(mixed_stfts+'/'+stft, header=None)
                mask = pd.read_csv(ground_truth_masks+'/'+mask, header=None)

                mask_tanh = np.tanh(0.1*mask)
                
                yield (np.transpose(stft), np.transpose(mask_tanh))

                i+=1

# input shape will be (batch_size, time_length, 129)
class RNN(keras.models.Model):
    def __init__(self):
        super().__init__()
        self._build_model()
        super().__init__(self._x,self._y)
        self.compile(loss='mse', optimizer='adam')

    def _build_model(self):
        x = layers.Input(shape=(None, 129))
        h = layers.LSTM(128, return_sequences=True)(x)
        h = layers.TimeDistributed(layers.Dense(128, activation='relu'))(h)
        y = layers.TimeDistributed(layers.Dense(129, activation='tanh'))(h)

        self._x = x
        self._y = y

class Machine():
    def __init__(self, mixed_stfts, ground_truth_masks, val_mixed_stfts, val_ground_truth_masks, batch_size=128, val_batch_size=32, shuffle=False, init=False, flat=False, clean_files=None, mixed_files=None, val_clean_files=None, val_mixed_files=None):
        self.set_data(mixed_stfts, ground_truth_masks, val_mixed_stfts, val_ground_truth_masks, batch_size=batch_size, val_batch_size=val_batch_size, shuffle=shuffle, init=init, flat=flat, clean_files=clean_files, mixed_files=mixed_files, val_clean_files=val_clean_files, val_mixed_files=val_mixed_files)
        self.set_model()
        self.exp = 0

    def set_data(self, mixed_stfts, ground_truth_masks, val_mixed_stfts, val_ground_truth_masks, batch_size, val_batch_size, shuffle, init, flat, clean_files, mixed_files, val_clean_files, val_mixed_files):
        self.data = Dataset(mixed_stfts, ground_truth_masks, batch_size=batch_size, shuffle=shuffle, init=init, flat=flat, clean_files=clean_files, mixed_files=mixed_files)
        self.val_data = Dataset(val_mixed_stfts, val_ground_truth_masks, batch_size=val_batch_size, shuffle=shuffle, init=init, flat=flat, clean_files=val_clean_files, mixed_files=val_mixed_files)
    
    def set_model(self):
        self.model = RNN()

    def load_model(self, checkpoint):
        self.model.load_weights(checkpoint)
        print("### Successfully loaded weights from {checkpoint} ###".format(checkpoint=checkpoint))
    
    def load_latest(self):
        exp = 0
        latest = tf.train.latest_checkpoint('ckpt')
        try:
            exp = int(latest[8:11]) + int(latest[12:14])
        except:
            exp = int(latest[8:11])
        self.load_model(latest)
        self.exp = exp

    class historyLogger(tf.keras.callbacks.Callback):
        def __init__(self, filepath):
            super().__init__()
            self.filepath = filepath

        def on_epoch_end(self, epoch, logs):
            df = DataFrame(logs, index=[0])
            df.to_csv(self.filepath, header=False, index=False, mode='a')

    def fit(self, epochs=10, verbose=1, save=True, hist_save=True):
        data = self.data
        val_data = self.val_data
        model = self.model
        exp = self.exp
        callbacks=[]
        if save:
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='ckpt/cp_{exp:03d}+{{epoch:02d}}.cpkt'.format(exp=exp), monitor='loss', verbose=1, save_weights_only=True, save_best_only=False)
            callbacks.append(cp_callback)
        if hist_save:
            hist_callback = self.historyLogger("history.csv")
            callbacks.append(hist_callback)
        history = model.fit(data.dataset, epochs=epochs, verbose=verbose, callbacks=callbacks, validation_data=val_data.dataset)
        return history
    
    def plot(self, history):
        plt.figure(0, figsize=(10,6))
        plt.subplot(211)
        plt.plot(history.history['accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.subplot(212)
        plt.plot(history.history['loss'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')

    def estimate(self, input_path, output_path):
        input_data = librosa.load(input_path, sr=16000, dtype='float64')[0]
        input_stft = librosa.stft(input_data, n_fft=256, hop_length=128, win_length=256)
        
        input_stft_mag = np.abs(input_stft) # (129, timestep)
        input_stft_phase = np.angle(input_stft)

        cIRM_mag_tanh = np.transpose(np.squeeze(self.model(np.expand_dims(np.transpose(input_stft_mag), 0), training=None), axis=0)) # (129, timestep)
        cIRM_mag = 10 * np.arctanh(cIRM_mag_tanh)
        est_stft_mag = cIRM_mag * input_stft_mag
        est_stft_phase = input_stft_phase
        est_stft = est_stft_mag * np.exp(1j*est_stft_phase)
        
        est_data = librosa.istft(est_stft, hop_length=128, win_length=256)
        est_data = est_data*2**15 # for transition between librosa and wave

        output_file = wave.Wave_write(output_path)
        output_file.setparams(wave.open(input_path, "r").getparams()) #nchannels, sampwidth, framerate, nframes, comptype, compname
        output_file.writeframes(array.array('h', est_data.astype(np.int16)).tobytes())
        output_file.close()

    def run(self, epochs=10, verbose=1, save=True, hist_save=True, plot=True, from_ckpt=True):
        if from_ckpt:
            self.load_latest()
        history = self.fit(epochs=epochs, verbose=verbose, save=save, hist_save=hist_save)
        if plot:
            self.plot(history)

if __name__=='__main__':
    mach = Machine(mixed_stfts='D:/RnE/data/mixed_stft/TRAIN', ground_truth_masks='D:/RnE/data/ground_truth_mask/TRAIN', val_mixed_stfts='D:/RnE/data/mixed_stft/TEST', val_ground_truth_masks='D:/RnE/data/ground_truth_mask/TEST', batch_size=128, val_batch_size=32, shuffle=False, init=False, flat=False, clean_files=None, mixed_files=None, val_clean_files=None, val_mixed_files=None)
    mach.load_latest()
    mach.estimate(input_path='D:/RnE/data/mixed_data/TEST/SI458.WAV.wav+DKITCHEN_ch01.wav--snr0.wav', output_path='D:/RnE/data/estimated/TEST/SI458.WAV.wav+DKITCHEN_ch01.wav--snr0.wav')
    mach.estimate(input_path='D:/RnE/data/mixed_data/TEST/SI458.WAV.wav+NFIELD_ch05.wav--snr0.wav', output_path='D:/RnE/data/estimated/TEST/SI458.WAV.wav+NFIELD_ch05.wav--snr0.wav')
    mach.estimate(input_path='D:/RnE/data/mixed_data/TEST/SI458.WAV.wav+DLIVING_ch01.wav--snr0.wav', output_path='D:/RnE/data/estimated/TEST/SI458.WAV.wav+DLIVING_ch01.wav--snr0.wav')
    mach.estimate(input_path='D:/RnE/data/mixed_data/TEST/SI458.WAV.wav+NPARK_ch01.wav--snr0.wav', output_path='D:/RnE/data/estimated/TEST/SI458.WAV.wav+DLIVING_ch01.wav--snr0.wav')