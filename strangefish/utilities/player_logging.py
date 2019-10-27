import logging
import sys
import os

TAGS = [
    '\N{four leaf clover}',
    '\N{skull}',
    '\N{bacon}',
    '\N{spouting whale}',
    '\N{fire}',
    '\N{eagle}',
    '\N{smiling face with sunglasses}',
    '\N{beer mug}',
    '\N{rocket}',
    '\N{snake}',
    '\N{butterfly}',
    '\N{jack-o-lantern}',
    '\N{white medium star}',
    '\N{hot beverage}',
    '\N{earth globe americas}',
    '\N{red apple}',
    '\N{robot face}',
    '\N{sunflower}',
    '\N{doughnut}',
    '\N{crab}',
    '\N{soccer ball}',
    '\N{hibiscus}',
]

concise_format = '%(process)-5d %(asctime)8s  {}  %(message)s'
verbose_format = logging.Formatter('%(name)s - %(process)d - %(asctime)s - '
                                   '%(levelname)s from %(module)s.%(funcName)s, line %(lineno)d: '
                                   '"%(message)s"')

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'gameLogs')
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)


def create_main_logger(log_to_file: bool = False) -> logging.Logger:
    logger = logging.getLogger(name=f"strangefish.{os.getpid()}")

    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        stdout_handler.setFormatter(logging.Formatter(concise_format.format(TAGS[os.getpid() % len(TAGS)]), "%I:%M:%S"))
        logger.addHandler(stdout_handler)

    if log_to_file and len(logger.handlers) < 2:
        file_handler = logging.FileHandler(os.path.join(LOG_DIR, f'StrangeFish.log'))
        file_handler.setFormatter(verbose_format)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

    return logger


def create_sub_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name=f"strangefish.{os.getpid()}.{name}")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger
