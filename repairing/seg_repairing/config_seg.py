cfg = dict()
    
cfg["batch_size"] = 2
cfg["n_epochs"] = 100
cfg["drop_rate"] = 0.0

cfg['data_path'] = '../../data/Alan'

cfg["num_workers"] = 4

cfg["resolution"] = 64

cfg["lr"] = 0.001
cfg["wd"] = 0.0001
cfg["momentum"] = 0.9

cfg["bg_loss"] = 0.1
cfg["focal_gamma"] = 2


cfg["milestones"] = [25, 40]
cfg["gamma"] = 0.1

cfg["save_all"] = False
cfg["use_syncbn"] = True

cfg["base_path"] = "./checkpoints/alan"
cfg["run_flag"] = "resnet_"
