import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from create_datasets.create_data import create_data

class Dataset():
    def __init__(self, mixed_stfts, ground_truth_masks, shuffle=False, batch_size=1, init=False, flat=True, clean_files=None, mixed_files=None):
        self.mixed_stfts = mixed_stfts
        self.ground_truth_masks = ground_truth_masks
        self.flat = flat
        #if init:
            #create_data(clean_files=clean_files, mixed_files=mixed_files, output_ground_truth_masks=ground_truth_masks, output_mixed_stfts=mixed_stfts)
        self.dataset = tf.data.Dataset.from_generator(self._gen, output_signature=(tf.TensorSpec(shape=(None, 129), dtype=np.float32), tf.TensorSpec(shape=(None, 129), dtype=np.float32)))
        self.dataset = self.dataset.padded_batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        if shuffle:
            self.dataset = self.dataset.shuffle(20000, reshuffle_each_iteration=True)
        
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
        self.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    def _build_model(self):
        x = layers.Input(shape=(None, 129))
        h = layers.LSTM(128, return_sequences=True)(x)
        h = layers.TimeDistributed(layers.Dense(128, activation='relu'))(h)
        y = layers.TimeDistributed(layers.Dense(129, activation='tanh'))(h)

        self._x = x
        self._y = y

class Machine():
    def __init__(self, mixed_stfts, ground_truth_masks, batch_size=128, shuffle=False, init=False, flat=False, clean_files=None, mixed_files=None):
        self.set_data(mixed_stfts, ground_truth_masks, batch_size=batch_size, shuffle=shuffle, init=init, flat=flat, clean_files=clean_files, mixed_files=mixed_files)
        self.set_model()
    
    def set_data(self, mixed_stfts, ground_truth_masks, batch_size, shuffle, init, flat, clean_files, mixed_files):
        self.data = Dataset(mixed_stfts, ground_truth_masks, batch_size=batch_size, shuffle=shuffle, init=init, flat=flat, clean_files=clean_files, mixed_files=mixed_files)

    def set_model(self):
        self.model = RNN()

    def load_model(self, checkpoint):
        self.model.load_weights(checkpoint)
        print("### Successfully loaded weights from {checkpoint} ###".format(checkpoint=checkpoint))
    
    def fit(self, epochs=10, verbose=1, save=True, exp=0):
        data = self.data
        model = self.model
        if save:
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='ckpt/cp_{exp:03d}+{{epoch:02d}}.cpkt'.format(exp=exp), monitor='loss', verbose=1, save_weights_only=True, save_best_only=False)
            history = model.fit(data.dataset, epochs=epochs, verbose=verbose, callbacks=[cp_callback])
        else:
            history = model.fit(data.dataset, epochs=epochs, verbose=verbose)
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

    def run(self, epochs=10, verbose=1, save=True, plot=True, from_ckpt=True):
        exp=0
        if from_ckpt:
            latest = tf.train.latest_checkpoint('ckpt')
            try:
                exp = int(latest[8:11]) + int(latest[12:14])
            except:
                exp = int(latest[8:11])
            self.load_model(latest)
        history = self.fit(epochs=epochs, verbose=verbose, save=save, exp=exp)
        if plot:
            self.plot(history)
if __name__=='__main__':
    mach = Machine('D:/RnE/data/mixed_stft', 'D:/RnE/data/ground_truth_mask', batch_size=128, shuffle=False, init=False, flat=False)
    mach.run(epochs=50, verbose=1, save=True, plot=True, from_ckpt=True)