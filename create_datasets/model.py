import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import os
import pandas as pd
import numpy as np
import librosa

class Dataset():
    def __init__(self, mixed_data, ground_truth_mask, Flatten=True):
        self.mixed_data = mixed_data
        self.ground_truth_mask = ground_truth_mask
        self.Flatten = Flatten
        self.create_dataset()

    def data_generator(self):
        mixed_data = self.mixed_data
        ground_truth_mask = self.ground_truth_mask

        masks = os.listdir(ground_truth_mask)
        i = 0
        while True:
            if i >= len(masks):
                break
            
            mask_imag = masks[i]
            mask_real = masks[i+1]
                
            mask_imag_DF = pd.read_csv(ground_truth_mask + '/'+ mask_imag, header=None)
            mask_imag_np = np.array(mask_imag_DF)
            mask_real_DF = pd.read_csv(ground_truth_mask + '/'+ mask_real, header=None)
            mask_real_np = np.array(mask_real_DF)


            mixed = mixed_data + '/' + mask_imag[:-21]
            mixed_np = librosa.load(mixed, sr=16000, dtype="float64")[0]
            mixed_stft = librosa.stft(mixed_np, n_fft=128, hop_length=64, win_length=128)
            mixed_real = np.real(mixed_stft)
            mixed_imag = np.imag(mixed_stft)
            if self.Flatten:
                yield (np.array([mixed_real, mixed_imag]).flatten(), np.array([mask_real_np, mask_imag_np]).flatten())
                i+=1
            else:
                yield (np.array([mixed_real, mixed_imag]), np.array([mask_real_np, mask_imag_np]))
                i+=1

    def create_dataset(self):
        dataset = tf.data.Dataset.from_generator(self.data_generator, output_types=(tf.float64, tf.float64), output_shapes=((98670,), (98670,)))
        dataset = dataset.cache('C:/Users/aquamj/Documents/GitHub/RnE/create_datasets/cache')
        dataset = dataset.prefetch(2)
        self.dataset = dataset

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

class Machine():
    def __init__(self, mixed_data, ground_truth_mask):
        self.mixed_data = mixed_data
        self.ground_truth_mask = ground_truth_mask
        self.set_data()
        self.set_model()
    
    def set_data(self):
        mixed_data = self.mixed_data
        ground_truth_mask = self.ground_truth_mask

        self.data = Dataset(mixed_data, ground_truth_mask)
        for mixed_data, ground_truth_mask in self.data.dataset:
            self.in_shape = mixed_data.shape
            self.out_shape = ground_truth_mask.shape
            break
    
    def set_model(self):
        in_shape = self.in_shape
        out_shape = self.out_shape
        self.model = DNN(in_shape, out_shape)
    
    def fit(self, epochs=10, batch_size=128, verbose=1):
        data = self.data
        model = self.model

        history = model.fit(data.dataset, epochs=epochs, batch_size=batch_size, verbose=verbose)
        return history

    def run(self, epochs=10, batch_size=128, verbose=1):
        self.fit(epochs=epochs, batch_size=batch_size, verbose=verbose)

machine = Machine('C:/Users/aquamj/Documents/GitHub/RnE/create_datasets/data/mixed_data', 'C:/Users/aquamj/Documents/GitHub/RnE/create_datasets/data/ground_truth_mask')
machine.run()
