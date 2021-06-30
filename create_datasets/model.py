import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import os
import pandas as pd
import numpy as np
import librosa

class Dataset():
    def __init__(self, mixed_stfts, ground_truth_masks, shuffle=False, batch_size=1):
        self.mixed_stfts = mixed_stfts
        self.ground_truth_masks = ground_truth_masks
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset = tf.data.Dataset.from_generator(self._gen, output_signature=(tf.TensorSpec(shape=(None, 129), dtype=np.float32), tf.TensorSpec(shape=(None, 129), dtype=np.float32)))
        if self.shuffle:
            self.dataset = self.dataset.shuffle(20000)
        self.dataset = self.dataset.batch(batch_size)

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

class DNN(keras.models.Model):
    def __init__(self, in_shape, out_shape):
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.build_model()
        super().__init__(self.x, self.y)
        self.compile_model()

    def build_model(self):
        in_shape = self.in_shape
        out_shape = self.out_shape
        x = layers.Input(in_shape)
        h = layers.Dense(128, activation='relu', name='e')(x)
        h = layers.Dropout(0.25)(h)
        h = layers.Dense(64, activation='relu')(h)
        h = layers.Dropout(0.5)(h)
        y = layers.Dense(out_shape[0], activation='relu', name='d')(h)

        self.x = x
        self.y = y

    def compile_model(self):
        self.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# input shape will be (batch_size, 129, time_length)
class RNN(keras.models.Model):
    def __init__(self):
        super().__init__()
        self._build_model()
        super().__init__(self._x,self._y)
        self.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    def _build_model(self):
        x = layers.Input(shape=(None, 129))
        y = layers.LSTM(129, return_sequences=True)(x)
        self._x = x
        self._y = y

class Machine():
    def __init__(self, mixed_stfts, ground_truth_masks, batch_size=128, shuffle=False):
        self.mixed_stfts = mixed_stfts
        self.ground_truth_masks = ground_truth_masks
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.set_data()
        self.set_model()
    
    def set_data(self):
        mixed_stfts = self.mixed_stfts
        ground_truth_masks = self.ground_truth_masks
        batch_size = self.batch_size
        shuffle = self.shuffle

        self.data = Dataset(mixed_stfts, ground_truth_masks, batch_size=batch_size, shuffle=shuffle)

        for i, j in self.data.dataset:
            print(i.shape)
            print(j.shape)
            break

    def set_model(self):
        self.model = RNN()
    
    def fit(self, epochs=10, verbose=1):
        data = self.data
        model = self.model

        history = model.fit(data.dataset, epochs=epochs, verbose=verbose)
        return history

    def run(self, epochs=10, verbose=1):
        self.fit(epochs=epochs, verbose=verbose)


mach = Machine('C:/Users/aquamj/Documents/GitHub/RnE/create_datasets/data/mixed_stft', 'C:/Users/aquamj/Documents/GitHub/RnE/create_datasets/data/ground_truth_mask', batch_size=128, shuffle=False)
mach.run(epochs=10, verbose=1)