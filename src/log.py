import datetime
import json
import logging
import logging.config
import logging.handlers
import weakref
from pathlib import Path
from typing import ClassVar, Optional, Self

RESULT_DATA_PREFIX = "_data"

__all__ = [
    "configure_logging",
    "get_default_log_config",
    "get_logger",
    "get_result_logger",
    "log_unexpected_exception",
]


def get_result_logger(logger_name: Optional[str] = None):
    return ResultLogger(logger_name)


def get_logger(name: Optional[str] = None):
    return logging.getLogger(name)


def log_unexpected_exception():
    """
    Enable logging for unexpected exceptions.
    """
    import sys

    def catch_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        logging.error(  # noqa: LOG015
            "Unexpected exception", exc_info=(exc_type, exc_value, exc_traceback)
        )

    sys.excepthook = catch_exception


def configure_logging(log_config: dict):
    logging.config.dictConfig(log_config)


def get_default_log_config(log_dir: Path, file_name: str):
    if not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)

    if not log_dir.is_dir():
        raise ValueError(f"{log_dir} is not a directory")

    log_file_path = log_dir / f"{file_name}.log"
    result_file_path = log_dir / f"{file_name}.jsonl"

    return {
        "version": 1,
        "formatters": {
            "iso": {"()": f"{__name__}.ISOFormatter"},
            "simple": {
                "datefmt": "%H:%M:%S",
                "format": "%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "simple",
                "level": "INFO",
                "stream": "ext://sys.stderr",
            },
            "file": {
                "()": f"{__name__}.FileHandler",
                "file_path": log_file_path,
                "formatter": "iso",
                "level": "DEBUG",
            },
            "queue_handler": {
                "class": "logging.handlers.QueueHandler",
                "handlers": ["console", "file"],
                "listener": f"{__name__}.QueueListener",
                "queue": {"()": "multiprocessing.Queue", "maxsize": -1},
                "respect_handler_level": True,
            },
            "result": {
                "()": f"{__name__}.ResultHandler",
                "file_path": result_file_path,
            },
            "result_queue_handler": {
                "class": "logging.handlers.QueueHandler",
                "handlers": ["result"],
                "listener": f"{__name__}.QueueListener",
                "queue": {"()": "multiprocessing.Queue", "maxsize": -1},
            },
        },
        "loggers": {
            "custom_log_result": {
                "handlers": ["result_queue_handler"],
                "level": "INFO",
                "propagate": False,
            },
            "root": {"handlers": ["queue_handler"], "level": "DEBUG"},
        },
    }


def _formatTime(record, datefmt=None):
    return (
        datetime.datetime.fromtimestamp(record.created, datetime.timezone.utc)
        .astimezone()
        .isoformat(sep="T", timespec="milliseconds")
    )


class ISOFormatter(logging.Formatter):
    def __init__(self, fmt=None):
        if fmt is None:
            fmt = "%(asctime)s %(levelname)-8s[%(filename)s:%(funcName)s:%(lineno)d] %(message)s"  # noqa: E501

        super().__init__(fmt=fmt, datefmt=None)

    def formatTime(self, record, datefmt=None):
        return _formatTime(record, datefmt)


class FileHandler(logging.FileHandler):
    def __init__(
        self,
        file_path: Path,
        mode: str = "w",
        encoding: Optional[str] = None,
        delay: bool = False,
        errors: Optional[str] = None,
    ) -> None:
        if not file_path.parent.exists():
            file_path.parent.mkdir(parents=True)

        super().__init__(file_path, mode, encoding, delay, errors)


class ResultHandler(logging.Handler):
    def __init__(self, file_path: Path):
        super().__init__()
        self.file_path = file_path

        # Create the file if it doesn't exist
        open(self.file_path, "w").close()
        self.file = open(self.file_path, "a")

    def emit(self, record):
        asctime = _formatTime(record)
        log_entry = {
            "timestamp": asctime,
        }

        if hasattr(record, RESULT_DATA_PREFIX):
            log_entry.update(getattr(record, RESULT_DATA_PREFIX))

        try:
            json.dump(log_entry, self.file)
            self.file.write("\n")
        except Exception:
            self.handleError(record)

    def __del__(self):
        self.file.close()

    def close(self):
        try:
            if hasattr(self, "file") and not self.file.closed:
                self.file.close()
        finally:
            super().close()


class QueueListener(logging.handlers.QueueListener):
    _instances: ClassVar[set[Self]] = set()

    def __init__(self, queue, *handlers, respect_handler_level=False) -> None:
        super().__init__(queue, *handlers, respect_handler_level=respect_handler_level)
        self.start()
        self._finalizer = weakref.finalize(self, self._cleanup)
        self.__class__._instances.add(self)

    def _cleanup(self):
        try:
            self.stop()
            self.queue.join()  # type: ignore
        except Exception:  # noqa: S110
            pass
        finally:
            self.__class__._instances.discard(self)


class ResultLogger:
    def __init__(self, logger_name: Optional[str] = None):
        logger_name = logger_name if logger_name else ""
        self.logger = logging.getLogger(f"custom_log_result.{logger_name}")

    def log(self, data: dict):
        self.logger.info("", extra={RESULT_DATA_PREFIX: data})
