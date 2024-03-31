import logging
import os
import sys


def create_logger():
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] "
        "[%(filename)s:%(lineno)d:%(funcName)s] %(message)s"
    )

    new_logger = logging.getLogger("FlexTrain")
    new_logger.setLevel(logging.INFO)
    new_logger.propagate = False

    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    new_logger.addHandler(sh)

    return new_logger


logger = create_logger()


def rank0_info(*args, **kwargs):
    if int(os.getenv('RANK', '0')) == 0:
        logger.info(*args, **kwargs)


def print_configuration(args, name):
    logger.info("{}:".format(name))
    for arg in sorted(vars(args)):
        dots = "." * (29 - len(arg))
        logger.info("  {} {} {}".format(arg, dots, getattr(args, arg)))
