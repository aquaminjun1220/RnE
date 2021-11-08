from model.opt import opt_no_phase_Model

if __name__=="__main__":
    opt_no_phase = opt_no_phase_Model(clean_file_paths="./data\clean_data\TIMIT\TRAIN", mixed_file_paths="D:\RnE\data\mixed_data\TRAIN")

    opt_no_phase.run(num_data=-1, output_paths="D:\RnE\data\opt\TRAIN")