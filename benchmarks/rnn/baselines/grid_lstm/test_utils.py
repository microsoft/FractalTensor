import tensorflow as tf


def get_config():
    config = tf.compat.v1.ConfigProto(
        gpu_options=tf.compat.v1.GPUOptions(
            allow_growth=True, per_process_gpu_memory_fraction=0.2))

    config.log_device_placement = False
    config.allow_soft_placement = True

    config.intra_op_parallelism_threads = 0
    config.inter_op_parallelism_threads = 56

    return config


def device(dtype="cpu"):
    """Return the TF device string.

    Args:
        dtype: String, "cpu" or "gpu".

    Raises:
        ValueError: if dtype is an unknown device.
    """
    if dtype == "cpu":
        return "/device:CPU:0"
    elif dtype == "gpu":
        assert tf.test.is_gpu_available(cuda_only=True)
        return "/device:GPU:0"
    else:
        raise ValueError("Unknown device type. Should be cpu or gpu.")
