from prep import prep
import json

with open("./json/prep_for_final copy.json", "r") as params_json:
    params = json.load(params_json)

for param in params:
    prep(clean_files=param["clean_files"], noise_files=param["noise_files"], output_mixed_files=param["output_mixed_files"], snrs=param["snrs"], output_ground_truth_masks=param["output_ground_truth_masks"], output_mixed_stfts=param["output_mixed_stfts"], num_clean=param["num_clean"], num_noise=param["num_noise"])