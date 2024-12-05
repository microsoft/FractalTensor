import logging


def get_logger(log_file_name='tvm_codegen.txt', log_level=logging.DEBUG):
    logger = logging.getLogger()
    logger.setLevel(log_level)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    fh = logging.FileHandler(log_file_name)
    fh.setLevel(log_level)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger
