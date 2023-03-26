import json
import os
import shutil
from typing import List

from loguru import logger
from pydantic import BaseModel

from . import models

logger.configure(handlers=[])


def setup_logging(logdir):
    try:
        shutil.rmtree(logdir + "/*")
    except FileNotFoundError:
        pass

    logger.add(
        os.path.join(logdir, "events.{time}.log"),
        rotation="1 hours",
        retention="2 hours",
        level="INFO",
        filter=__name__,
        enqueue=True,
        delay=True,
        serialize=True,
    )


EVENTS_LOG_DIR = os.environ.get("EVENTS_LOG_DIR", None)
if EVENTS_LOG_DIR is not None:
    setup_logging(EVENTS_LOG_DIR)


def log_completions(
    request: models.CompletionRequest, response: models.CompletionResponse
) -> None:
    event = models.CompletionEvent.build(request, response)
    logger.info(event.json())


def log_view(id: str, index: int) -> None:
    event = models.ChoiceEvent.build_view(id, index)
    logger.info(event.json())


def log_selection(id: str, index: int) -> None:
    event = models.ChoiceEvent.build_selection(id, index)
    logger.info(event.json())
