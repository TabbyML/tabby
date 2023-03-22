import logging
import sys
from dataclasses import asdict, dataclass

import models
from pythonjsonlogger import jsonlogger


def make_logger():
    jsonHandler = logging.StreamHandler(sys.stdout)
    jsonHandler.setFormatter(jsonlogger.JsonFormatter())

    logger = logging.getLogger("events")
    logger.setLevel(logging.INFO)
    logger.addHandler(jsonHandler)
    return logger


logger = make_logger()


@dataclass
class CompletionsEvent:
    prompt: str
    choices: models.Choice


def log_completions(
    request: models.CompletionsRequest, response: models.CompletionsResponse
) -> None:
    event = CompletionsEvent(prompt=request.prompt, choices=response.choices)
    logger.info(asdict(event))
