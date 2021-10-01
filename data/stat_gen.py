import wave
import numpy as np
import array

def stat_gen(mean, variance, length, output_folder):
    params = (1, 2, 16000, 16000*length, 'NONE', 'not compressed')
    output_path = output_folder + "/static" + "_" + str(mean) + "_" + str(variance) + ".wav"
    arr = np.random.normal(mean, variance, 16000*length)
    wf = wave.Wave_write(output_path)
    wf.setparams(params)
    wf.writeframes(array.array('h', arr.astype(np.int16)).tobytes())
    wf.close()

def stat_gen_from_list(meanvar, length, output_folder):
    for mean, var in meanvar:
        stat_gen(mean, var, length, output_folder)

if __name__=="__main__":
    meanvar = [(0, 10**i) for i in range(5)]
    stat_gen_from_list(meanvar, 60, "data/noise_data/stat_noise")