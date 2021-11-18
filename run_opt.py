from model.opt import opt_Model, opt_no_phase_Model
from check.measure_SNR_LSD import comp_SNR, comp_LSD3
import json
import os
import librosa
from pesq import pesq

if __name__=="__main__":
	opt = opt_Model()
	opt_no_phase = opt_no_phase_Model()
	with open("./json/run_opt.json", 'r') as json_file:
		params = json.load(json_file)
	res = dict()
	res["opt"] = dict()
	res["opt_no_phase"] = dict()
	for ress in res.values():
		ress["nonstationary"] = dict()
		ress["stationary"] = dict()
	errors = params[0]["errors"]
	for ress in res.values():
			for resss in ress.values():
				for error in errors:
					resss["error:"+str(error)] = dict()
	est_time = 0
	for param in params:
		clean_file_paths = param["clean_file_paths"]
		mixed_file_paths = param["mixed_file_paths"]
		noise_type = param["noise_type"]
		errors = param["errors"]
		for mixed_file_path in os.listdir(mixed_file_paths)[:1200]:
			clean_file_path = mixed_file_path.split("+")[0]
			snr = mixed_file_path[mixed_file_path.index("snr")+3 : -4]
			mixed_data = librosa.load(mixed_file_paths+mixed_file_path, sr=16000, dtype='float64')[0]
			clean_data = librosa.load(clean_file_paths+clean_file_path, sr=16000, dtype='float64')[0]

			for error in errors:
				est_data = opt.estimate(clean_data=clean_data, mixed_data=mixed_data, error=error)
				est_time += 1
				PESQw = pesq(16000, ref=clean_data*2**15, deg=est_data*2**15, mode="wb")
				PESQn = pesq(16000, ref=clean_data*2**15, deg=est_data*2**15, mode="nb")

				if "snr:"+snr not in res["opt"][noise_type]["error:"+str(error)].keys():
					res["opt"][noise_type]["error:"+str(error)]["snr:"+snr] = dict()
				if "time" not in res["opt"][noise_type]["error:"+str(error)]["snr:"+snr].keys():
					res["opt"][noise_type]["error:"+str(error)]["snr:"+snr]["time"] = 0
				if "PESQw" not in res["opt"][noise_type]["error:"+str(error)]["snr:"+snr].keys():
					res["opt"][noise_type]["error:"+str(error)]["snr:"+snr]["PESQw"] = 0
				if "PESQn" not in res["opt"][noise_type]["error:"+str(error)]["snr:"+snr].keys():
					res["opt"][noise_type]["error:"+str(error)]["snr:"+snr]["PESQn"] = 0
				
				res["opt"][noise_type]["error:"+str(error)]["snr:"+snr]["PESQw"] += PESQw
				res["opt"][noise_type]["error:"+str(error)]["snr:"+snr]["PESQn"] += PESQn
				res["opt"][noise_type]["error:"+str(error)]["snr:"+snr]["time"] += 1

				est_data2 = opt_no_phase.estimate(clean_data=clean_data, mixed_data=mixed_data, error=error)
				PESQw2 = pesq(16000, ref=clean_data*2**15, deg=est_data2*2**15, mode="wb")
				PESQn2 = pesq(16000, ref=clean_data*2**15, deg=est_data2*2**15, mode="nb")

				if "snr:"+snr not in res["opt_no_phase"][noise_type]["error:"+str(error)].keys():
					res["opt_no_phase"][noise_type]["error:"+str(error)]["snr:"+snr] = dict()
				if "time" not in res["opt_no_phase"][noise_type]["error:"+str(error)]["snr:"+snr].keys():
					res["opt_no_phase"][noise_type]["error:"+str(error)]["snr:"+snr]["time"] = 0
				if "PESQw" not in res["opt_no_phase"][noise_type]["error:"+str(error)]["snr:"+snr].keys():
					res["opt_no_phase"][noise_type]["error:"+str(error)]["snr:"+snr]["PESQw"] = 0
				if "PESQn" not in res["opt_no_phase"][noise_type]["error:"+str(error)]["snr:"+snr].keys():
					res["opt_no_phase"][noise_type]["error:"+str(error)]["snr:"+snr]["PESQn"] = 0
				
				res["opt_no_phase"][noise_type]["error:"+str(error)]["snr:"+snr]["PESQw"] += PESQw2
				res["opt_no_phase"][noise_type]["error:"+str(error)]["snr:"+snr]["PESQn"] += PESQn2
				res["opt_no_phase"][noise_type]["error:"+str(error)]["snr:"+snr]["time"] += 1
				if est_time%100==0:
					print("est_time:", est_time)
				
				if est_time%200==0:
					with open("./json/res_opt_PESQ.json", "w") as json_res:
						json.dump(res, json_res, indent=4, sort_keys=True)
						print("saved")

	for ress in res.values():
		for resss in ress.values():
			for ressss in resss.values():
				for resssss in ressss.values():
					resssss["PESQw"] = resssss["PESQw"]/resssss["time"]
					resssss["PESQn"] = resssss["PESQn"]/resssss["time"]


	with open("./json/res_opt_PESQ.json", "w") as json_res:
		json.dump(res, json_res, indent=4, sort_keys=True)



