config = {
    "data": {
        "data_dir": "./data/ICEWS14_forecasting",
        "dataset": "ICEWS14_forecasting"
    },
    "training": {
        "state": "train",  # "train" or "test"
        "epochs_conv": 100,
        "lr": 0.0001,
        "weight_decay_conv": 0.000001
    },
    "model": {
        "embedding_size": 200
    },
    "temporal": {
        "neg_ratio": 1
    },
    "device": {
        "device_type": "cpu"  # cpu, cuda
    },
    "experiment": {
        "save_models": True,
        "results_dir": "./results/bestmodel",
        "log_level": "INFO"
    },
    "ds_specific": {
        "ICEWS14_forecasting": {
            "his_len": 13
        },
        "ICEWS18": {
            "his_len": 10
        },
        "ICEWS0515_forecasting": {
            "his_len": 150
        }
    }
}
