cfg = dict()

cfg["base_path"] = "../checkpoints/Alan_sa_epoch_1500_220314_152239"

cfg["data_path"] = "../../../data/Alan"
cfg['appearance'] = True

cfg["n_evaluation_samples"] = None

cfg["eval_batch_size"] = 1
cfg["n_workers"] = 4
cfg["target_resolution"] = 128

cfg['l2_penalty_weight'] = 1e-2
