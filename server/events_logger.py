import logging
from dataclasses import dataclass

import models
from pythonjsonlogger import jsonlogger


def make_logger():
    jsonHandler = logging.StreamHandler()
    jsonHandler.setLevel(logging.INFO)
    jsonHandler.setFormatter(jsonlogger.JsonFormatter())

    logger = logging.getLogger("events")
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
    logger.info(CompletionsEvent(prompt=request.prompt, choices=response.choices))
