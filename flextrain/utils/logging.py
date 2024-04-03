import logging
import os


class DummyLogger:
    """ A dummy logger that does nothing when its methods are called. """
    def __getattribute__(self, __name: str):
        return lambda *args, **kwargs: None


def create_logger(name, level=logging.INFO):
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


logger = create_logger("FlexTrain", level=logging.INFO) \
    if int(os.getenv("RANK", "0")) == 0 else DummyLogger()


def print_configuration(args, name):
    logger.info("{}:".format(name))
    for arg in sorted(vars(args)):
        dots = "." * (29 - len(arg))
        logger.info("  {} {} {}".format(arg, dots, getattr(args, arg)))
