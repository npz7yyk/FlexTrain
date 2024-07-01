import functools
import logging
import os


LEFT_BRACE = """{"""
RIGHT_BRACE = """}"""


class _DummyLogger:
    """ A dummy logger that does nothing when its methods are called. """
    def __getattribute__(self, __name: str):
        return lambda *args, **kwargs: None

    def warning_once(self, *args, **kwargs):
        pass


def _create_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] "
            "[%(filename)s:%(lineno)d:%(funcName)s] %(message)s"
        )
        handler.setFormatter(formatter)
        logger.setLevel(level)
        logger.addHandler(handler)
    return logger


logger = _create_logger("FlexTrain", level=logging.INFO)
rank0_logger = logger if int(os.getenv("RANK", "0")) == 0 else _DummyLogger()


@functools.lru_cache(None)
def warning_once(*args, **kwargs):
    """
    Emit the warning with the same message only once
    """
    logger.warning(*args, **kwargs)


logger.warning_once = warning_once


def print_rank0(*args, **kwargs):
    if int(os.getenv("RANK", "0")) == 0:
        print(*args, **kwargs)


def log_configuration(args, name):
    rank0_logger.info("{}:".format(name))
    for arg in sorted(vars(args)):
        dots = "." * (29 - len(arg))
        rank0_logger.info("  {} {} {}".format(arg, dots, getattr(args, arg)))
