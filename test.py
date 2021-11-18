from model.model import Dataset, RNN, Machine

P = [["128_128_basic", 128, 128, 'relu', 'tanh', 'mse', 'adam'],
["64_128_basic", 64, 128, 'relu', 'tanh', 'mse', 'adam'],
["32_128_basic", 32, 128, 'relu', 'tanh', 'mse', 'adam'],
["64_256_basic", 64, 256, 'relu', 'tanh', 'mse', 'adam']]

if __name__=='__main__':
    for params in P:
        mach = Machine(LSTM_unit=params[1], dense1_unit=params[2], dense1_act=params[3], dense2_act=params[4], loss=params[5], optimizer=params[6])
        mach.load_latest(save_filepath="model\savedmodel/"+params[0]+'/ckpt')
        mach.estimate(input_path="D:\RnE\data\mixed_data\TEST_stat\SA1.WAV.wav+static_0_10.wav--snr10.0.wav", output_path="D:\RnE\data\estimated/TEST_stat/"+params[0]+'/SA1.WAV.wav+static_0_10.wav--snr10.0.wav')