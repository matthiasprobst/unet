"""Logger"""

import logging
import pathlib

import appdirs

name = 'unet'

_logdir = pathlib.Path(appdirs.user_log_dir(name))
logger = logging.getLogger(name)

if _logdir.exists():
    _logFolderMsg = f'{name} log folder available: {_logdir}'
else:
    _logdir.mkdir(parents=True)
    _logFolderMsg = f'{name} log folder created at {_logdir}'

logger.setLevel(logging.DEBUG)

_formatter = logging.Formatter(
    '%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d_%H:%M:%S')

_file_handler = logging.handlers.RotatingFileHandler(_logdir / f'{name}.log', maxBytes=int(5e6), backupCount=2)
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(_formatter)
