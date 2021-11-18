from model.model import Dataset, RNN, Machine

P = [["128_128_basic", 128, 128, 'relu', 'tanh', 'mse', 'adam'],
["64_128_basic", 64, 128, 'relu', 'tanh', 'mse', 'adam'],
["32_128_basic", 32, 128, 'relu', 'tanh', 'mse', 'adam'],
["64_256_basic", 64, 256, 'relu', 'tanh', 'mse', 'adam']]

if __name__=='__main__':
    for params in P:
        mach = Machine(LSTM_unit=params[1], dense1_unit=params[2], dense1_act=params[3], dense2_act=params[4], loss=params[5], optimizer=params[6])
        mach.set_data(mixed_stfts='D:/RnE/data/mixed_stft/TRAIN_stat', ground_truth_masks='D:/RnE/data/ground_truth_mask/TRAIN_stat', val_mixed_stfts='D:/RnE/data/mixed_stft/TEST_stat', val_ground_truth_masks='D:/RnE/data/ground_truth_mask/TEST_stat', num_data=3000, val_num_data=600, batch_size=128, val_batch_size=32, shuffle=False, init=False, flat=False, clean_files=None, mixed_files=None, val_clean_files=None, val_mixed_files=None)
        
        mach.run(epochs=30, verbose=1, save=True, save_filepath='./model/savedmodel/'+params[0]+'/ckpt', hist_filepath='./model/savedmodel/'+params[0], plot=False, from_ckpt=True)