from autonlp.flags import Flags

#####################
# Parameters
#####################

### MLflow Tracking
flags_dict_mlflow = {
    # use MLflow Tracking (save in "./mlruns")
    # dashboard : ($ mlflow ui --backend-store-uri ./mlruns)
    "apply_mlflow": False,
    # MLflow Experiment name
    "experiment_name": "Experiment"
}


flags = Flags().update(flags_dict_mlflow)