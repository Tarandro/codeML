from autonlp.flags import Flags

#####################
# Parameters
#####################

### Ray Tune Optimization
flags_dict_ray_tune = {
    # use Ray Tune library
    "apply_ray": False,
    # number of model to train in parallel
    "ray_max_model_parallel": 1,
    # number of cpu per model
    "ray_cpu_per_model": 1,
    # number of gpu per model
    "ray_gpu_per_model": 0,
    # verbose ray
    "ray_verbose": 2
}


flags = Flags().update(flags_dict_ray_tune)