import os
import shutil

from loguru import logger
from pydantic import BaseModel

from . import models


def setup_logging(logdir):
    try:
        shutil.rmtree(logdir)
    except FileNotFoundError:
        pass

    # Remove default handler
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


def log_completion(
    request: models.CompletionRequest, response: models.CompletionResponse
) -> None:
    event = models.CompletionEvent.build(request, response)
    logger.info(event.json())


def log_event(event: models.Event):
    logger.info(event.json())
