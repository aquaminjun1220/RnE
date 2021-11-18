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

"""mixed_stft, ground_truth_mask, shuffle, batch_size, flat, clean_files, mixed_files, init
"""
class Dataset():
    def __init__(self, mixed_stfts, ground_truth_masks, num_data=15000, shuffle=False, batch_size=1, flat=True, clean_files=None, mixed_files=None, init=False):
        self.mixed_stfts = mixed_stfts
        self.ground_truth_masks = ground_truth_masks
        self.flat = flat
        self.num_data = num_data
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
        num_data = self.num_data
        stfts = os.listdir(mixed_stfts)
        masks = os.listdir(ground_truth_masks)
        if flat:
            i = 0
            while i<min(len(stfts), num_data):
                stft = stfts[i]
                mask = masks[i]
                
                stft = pd.read_csv(mixed_stfts+'/'+stft, header=None)
                mask = pd.read_csv(ground_truth_masks+'/'+mask, header=None)
                
                yield (np.transpose(stft), np.transpose(mask))

                i+=1
        else:
            i = 0
            while i<min(len(stfts), num_data):
                stft = stfts[i]
                mask = masks[i]
                
                stft = pd.read_csv(mixed_stfts+'/'+stft, header=None)
                mask = pd.read_csv(ground_truth_masks+'/'+mask, header=None)

                mask_tanh = np.tanh(0.1*mask)
                
                yield (np.transpose(stft), np.transpose(mask_tanh))

                i+=1

# input shape will be (batch_size, time_length, 129)
class RNN(keras.models.Model):
    def __init__(self, LSTM_unit=128, dense1_unit=128, dense1_act='relu', dense2_act='tanh', loss='mse', optimizer='adam'):
        self._build_model(LSTM_unit=128, dense1_unit=128, dense1_act='relu', dense2_act='tanh')
        super().__init__(self._x,self._y)
        self.compile(loss=loss, optimizer=optimizer)

    def _build_model(self, LSTM_unit=128, dense1_unit=128, dense1_act='relu', dense2_act='tanh'):
        x = layers.Input(shape=(None, 129))
        h = layers.LSTM(LSTM_unit, return_sequences=True)(x)
        h = layers.TimeDistributed(layers.Dense(dense1_unit, activation=dense1_act))(h)
        y = layers.TimeDistributed(layers.Dense(129, activation=dense2_act))(h)

        self._x = x
        self._y = y

"""mixed_stfts, ground_truth_masks, val_mixed_stfts, val_ground_masks, batch_size, val_batch_size, 
shuffle, init, flat, clean_files, mixed_files, val_clean_files, val_mixed_files, """
class Machine():
    def __init__(self, LSTM_unit=128, dense1_unit=128, dense1_act='relu', dense2_act='tanh', loss='mse', optimizer='adam'):
        self.set_model(LSTM_unit=LSTM_unit, dense1_unit=dense1_unit, dense1_act=dense1_act, dense2_act=dense2_act, loss=loss, optimizer=optimizer)
        self.exp = 0

    def set_data(self, mixed_stfts, ground_truth_masks, val_mixed_stfts, val_ground_truth_masks, num_data, val_num_data, batch_size, val_batch_size, shuffle, init, flat, clean_files, mixed_files, val_clean_files, val_mixed_files):
        self.data = Dataset(mixed_stfts, ground_truth_masks, num_data=num_data, batch_size=batch_size, shuffle=shuffle, init=init, flat=flat, clean_files=clean_files, mixed_files=mixed_files)
        self.val_data = Dataset(val_mixed_stfts, val_ground_truth_masks, num_data=val_num_data, batch_size=val_batch_size, shuffle=shuffle, init=init, flat=flat, clean_files=val_clean_files, mixed_files=val_mixed_files)
    
    def set_model(self, LSTM_unit=128, dense1_unit=128, dense1_act='relu', dense2_act='tanh', loss='mse', optimizer='adam'):
        self.model = RNN(LSTM_unit=LSTM_unit, dense1_unit=dense1_unit, dense1_act=dense1_act, dense2_act=dense2_act, loss=loss, optimizer=optimizer)

    """checkpoint"""
    def load_model(self, checkpoint):
        self.model.load_weights(checkpoint)
        print("### Successfully loaded weights from {checkpoint} ###".format(checkpoint=checkpoint))
    
    def load_latest(self, save_filepath='./model/savedmodel/basic/ckpt'):
        exp = 0
        try:
            latest = tf.train.latest_checkpoint(save_filepath)
            latest_bas = os.path.basename(latest)
            exp = int(latest_bas[3:6]) + int(latest_bas[7:9])
            self.load_model(latest)
        except:
            print("### No previous checkpoints found at {save_filepath}!!  ###".format(save_filepath=save_filepath))
        self.exp = exp       

    class historyLogger(tf.keras.callbacks.Callback):
        def __init__(self, filepath='./model/savedmodel/basic/history.csv'):
            super().__init__()
            self.filepath = filepath

        def on_epoch_end(self, epoch, logs):
            df = DataFrame(logs, index=[0])
            df.to_csv(self.filepath, header=False, index=False, mode='a')

    def fit(self, epochs=10, verbose=1, save=True, save_filepath='./model/savedmodel/basic/ckpt', hist_save=True, hist_filepath='./model/savedmodel/basic'):
        data = self.data
        val_data = self.val_data
        model = self.model
        exp = self.exp
        callbacks=[]
        if save:
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=save_filepath+'/cp_{exp:03d}+{{epoch:02d}}.ckpt'.format(exp=exp), monitor='loss', verbose=1, save_weights_only=True, save_best_only=False)
            callbacks.append(cp_callback)
        if hist_save:
            hist_callback = self.historyLogger(hist_filepath+"/history.csv")
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

    def estimate(self, mixed_data):
        mixed_stft = librosa.stft(mixed_data, n_fft=256, hop_length=128, win_length=256)
        
        mixed_stft_mag = np.abs(mixed_stft) # (129, timestep)
        mixed_stft_phase = np.angle(mixed_stft)

        cIRM_mag_tanh = np.transpose(np.squeeze(self.model(np.expand_dims(np.transpose(mixed_stft_mag), 0), training=None), axis=0)) #  input (1, timestep,129) into model, squeeze & transpose output to (129, timestep)
        cIRM_mag = 10 * np.arctanh(cIRM_mag_tanh)
        est_stft_mag = cIRM_mag * mixed_stft_mag
        est_stft_phase = mixed_stft_phase
        est_stft = est_stft_mag * np.exp(1j*est_stft_phase)
        
        est_data = librosa.istft(est_stft, hop_length=128, win_length=256)
        return est_data

    def estimate_with_phase(self, mixed_data, clean_data):
        mixed_stft = librosa.stft(mixed_data, n_fft=256, hop_length=128, win_length=256)
        clean_stft = librosa.stft(clean_data, n_fft=256, hop_length=128, win_length=256)
        
        mixed_stft_mag = np.abs(mixed_stft) # (129, timestep)
        mixed_stft_phase = np.angle(clean_stft) # use clean stft phase

        cIRM_mag_tanh = np.transpose(np.squeeze(self.model(np.expand_dims(np.transpose(mixed_stft_mag), 0), training=None), axis=0)) #  mixed (1, timestep,129) into model, squeeze & transpose output to (129, timestep)
        cIRM_mag = 10 * np.arctanh(cIRM_mag_tanh)
        est_stft_mag = cIRM_mag * mixed_stft_mag
        est_stft_phase = mixed_stft_phase
        est_stft = est_stft_mag * np.exp(1j*est_stft_phase)
        
        est_data = librosa.istft(est_stft, hop_length=128, win_length=256)
        return est_data


    def run(self, epochs=10, verbose=1, save=True, save_filepath='./model/savedmodel/basic/ckpt', hist_save=True, hist_filepath='./model/savedmodel/basic', plot=True, from_ckpt=True):
        if from_ckpt:
            self.load_latest(save_filepath=save_filepath)
        exp = self.exp
        history = self.fit(epochs=epochs-exp, verbose=verbose, save=save, save_filepath=save_filepath, hist_save=hist_save, hist_filepath=hist_filepath)
        if plot:
            self.plot(history)

if __name__=='__main__':
    pass