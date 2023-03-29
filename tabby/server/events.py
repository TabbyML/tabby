import os
import shutil

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


LOGS_DIR = os.environ.get("LOGS_DIR", None)
if LOGS_DIR is not None:
    setup_logging(os.path.join(LOGS_DIR, "tabby-server"))


def log_completions(
    request: models.CompletionRequest, response: models.CompletionResponse
) -> None:
    event = models.CompletionEvent.build(request, response)
    logger.info(event.json())


def log_view(id: str, index: int) -> None:
    event = models.ChoiceEvent.build_view(id, index)
    logger.info(event.json())


def log_select(id: str, index: int) -> None:
    event = models.ChoiceEvent.build_select(id, index)
    logger.info(event.json())
