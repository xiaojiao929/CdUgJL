import logging
import os
from pathlib import Path


class Logger:
    def __init__(self, log_dir, name='train', filename='log.txt'):
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging.INFO)

        fmt = logging.Formatter('%(asctime)s %(levelname)s  %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

        fh = logging.FileHandler(os.path.join(log_dir, filename))
        fh.setFormatter(fmt)

        ch = logging.StreamHandler()
        ch.setFormatter(fmt)

        self._logger.addHandler(fh)
        self._logger.addHandler(ch)

        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=log_dir)
        except ImportError:
            self.writer = None

    def log(self, msg, level='info'):
        getattr(self._logger, level)(msg)

    def scalar(self, tag, value, step):
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def close(self):
        if self.writer is not None:
            self.writer.close()


def setup_logger(name, output='.', filename='log.txt'):
    return Logger(output, name=name, filename=filename)
