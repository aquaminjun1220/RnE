import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Dataset():
    def __init__(self, mixed_stfts, ground_truth_masks, shuffle=False, batch_size=1):
        self.mixed_stfts = mixed_stfts
        self.ground_truth_masks = ground_truth_masks
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset = tf.data.Dataset.from_generator(self._gen, output_signature=(tf.TensorSpec(shape=(None, 129), dtype=np.float32), tf.TensorSpec(shape=(None, 129), dtype=np.float32)))
        self.dataset = self.dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        if self.shuffle:
            self.dataset = self.dataset.shuffle(20000, reshuffle_each_iteration=True)
        

    def _gen(self):
        mixed_stfts = self.mixed_stfts
        ground_truth_masks = self.ground_truth_masks

        stfts = os.listdir(mixed_stfts)
        masks = os.listdir(ground_truth_masks)
        i = 0
        while True:
            if i >= len(stfts):
                break
            stft = stfts[i]
            mask = masks[i]
            
            stft = pd.read_csv(mixed_stfts+'/'+stft, header=None)
            mask = pd.read_csv(ground_truth_masks+'/'+mask, header=None)
            
            yield (np.transpose(stft), np.transpose(mask))

# input shape will be (batch_size, 129, time_length)
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
    def __init__(self, mixed_stfts, ground_truth_masks, batch_size=128, shuffle=False):
        self.set_data(mixed_stfts, ground_truth_masks, batch_size, shuffle)
        self.set_model()
    
    def set_data(self, mixed_stfts, ground_truth_masks, batch_size, shuffle):
        self.data = Dataset(mixed_stfts, ground_truth_masks, batch_size=batch_size, shuffle=shuffle)

    def set_model(self):
        self.model = RNN()
    
    def fit(self, epochs=10, verbose=1, save=True):
        data = self.data
        model = self.model
        if save:
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='ckpt/cp_{epoch:02d}.cpkt', monitor='loss', verbose=1, save_weights_only=True, save_best_only=True)
            history = model.fit(data.dataset, epochs=epochs, verbose=verbose, callbacks=[cp_callback], steps_per_epoch=117)
        else:
            history = model.fit(data.dataset, epochs=epochs, verbose=verbose, steps_per_epoch=117)
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

    def run(self, epochs=10, verbose=1, save=True, plot=True):
        history = self.fit(epochs=epochs, verbose=verbose, save=save)
        if plot:
            self.plot(history)

mach = Machine('D:/RnE/data/mixed_stft', 'D:/RnE/data/ground_truth_mask', batch_size=128, shuffle=False)
mach.run(epochs=100, verbose=1, save=True, plot=True)