cfg = dict()

cfg["base_path"] = "./checkpoints"
cfg["run_flag"] = "Alan_sa_epoch_1500_"

cfg['data_path'] = '../../data/Alan'

cfg["n_epochs"] = 1500
cfg["milestones"] = [150, 700, 1200]

# False for NeAR(S), True for NeAR(S+A)
cfg['appearance'] = True

cfg["training_resolution"] = 64
cfg["lr"] = 1e-3

cfg["batch_size"] = 8
cfg["eval_batch_size"] = 4
cfg["n_workers"] = 4
cfg["grid_noise"] = 0.01
cfg["uniform_grid_noise"] = True
cfg["target_resolution"] = 128
cfg["n_training_samples"] = None

cfg['l2_penalty_weight'] = 1e-2
