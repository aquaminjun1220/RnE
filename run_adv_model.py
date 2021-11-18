from model.model import Machine
import json
import os
import librosa
from check.measure_SNR_LSD import comp_SNR, comp_LSD3
from pesq import pesq

if __name__=="__main__":
	mach = Machine(LSTM_unit=128, dense1_unit=128, dense1_act='relu', dense2_act='tanh', loss='mse', optimizer='adam')
	mach.load_latest(save_filepath="model/savedmodel/basic/ckpt/")

	with open("./json/run_adv_model.json", 'r') as json_file:
		params = json.load(json_file)
	res = dict()
	res["model:basic"] = dict()
	res["model:advanced"] = dict()
	for ress in res.values():
		ress["nonstationary"] = dict()
		ress["stationary"] = dict()

	est_time=0
	for param in params:
		mixed_file_paths = param["mixed_file_paths"]
		clean_file_paths = param["clean_file_paths"]
		noise_type = param["noise_type"]
		for mixed_file_path in os.listdir(mixed_file_paths)[:1200]:
			clean_file_path = mixed_file_path.split("+")[0]
			snr = mixed_file_path[mixed_file_path.index("snr")+3 : -4]
			mixed_data = librosa.load(mixed_file_paths+mixed_file_path, sr=16000, dtype='float64')[0]
			clean_data = librosa.load(clean_file_paths+clean_file_path, sr=16000, dtype='float64')[0]

			est_data = mach.estimate(mixed_data)
			est_time+=1
			PESQw = pesq(16000, ref=clean_data*2**15, deg=est_data*2**15, mode="wb")
			PESQn = pesq(16000, ref=clean_data*2**15, deg=est_data*2**15, mode="nb")

			if "snr:"+snr not in res["model:basic"][noise_type].keys():
				res["model:basic"][noise_type]["snr:"+snr] = dict()
			if "time" not in res["model:basic"][noise_type]["snr:"+snr].keys():
				res["model:basic"][noise_type]["snr:"+snr]["time"] = 0
			if "PESQw" not in res["model:basic"][noise_type]["snr:"+snr].keys():
				res["model:basic"][noise_type]["snr:"+snr]["PESQw"] = 0
			if "PESQn" not in res["model:basic"][noise_type]["snr:"+snr].keys():
				res["model:basic"][noise_type]["snr:"+snr]["PESQn"] = 0

			res["model:basic"][noise_type]["snr:"+snr]["PESQw"] += PESQw
			res["model:basic"][noise_type]["snr:"+snr]["PESQn"] += PESQn
			res["model:basic"][noise_type]["snr:"+snr]["time"] += 1

			est_data2 = mach.estimate_with_phase(mixed_data, clean_data)
			PESQw2 = pesq(16000, ref=clean_data*2**15, deg=est_data2*2**15, mode="wb")
			PESQn2 = pesq(16000, ref=clean_data*2**15, deg=est_data2*2**15, mode="nb")

			if "snr:"+snr not in res["model:advanced"][noise_type].keys():
				res["model:advanced"][noise_type]["snr:"+snr] = dict()
			if "time" not in res["model:advanced"][noise_type]["snr:"+snr].keys():
				res["model:advanced"][noise_type]["snr:"+snr]["time"] = 0
			if "PESQw" not in res["model:advanced"][noise_type]["snr:"+snr].keys():
				res["model:advanced"][noise_type]["snr:"+snr]["PESQw"] = 0
			if "PESQn" not in res["model:advanced"][noise_type]["snr:"+snr].keys():
				res["model:advanced"][noise_type]["snr:"+snr]["PESQn"] = 0

			res["model:advanced"][noise_type]["snr:"+snr]["PESQw"] += PESQw2
			res["model:advanced"][noise_type]["snr:"+snr]["PESQn"] += PESQn2
			res["model:advanced"][noise_type]["snr:"+snr]["time"] += 1

			if est_time%100==0:
				print("est_time:", est_time)
				
			if est_time%200==0:
				with open("./json/res_adv_model_PESQ.json", "w") as json_res:
					json.dump(res, json_res, indent=4, sort_keys=True)
					print("saved")

	for ress in res.values():
		for resss in ress.values():
			for ressss in resss.values():
				ressss["PESQw"] = ressss["PESQw"]/ressss["time"]
				ressss["PESQn"] = ressss["PESQn"]/ressss["time"]

	with open("./json/res_adv_model_PESQ.json", "w") as json_res:
		json.dump(res, json_res, indent=4, sort_keys=True)