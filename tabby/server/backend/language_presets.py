from collections import defaultdict
from typing import List

from pydantic import BaseModel, Field

from ..models import Language


class LanguagePreset(BaseModel):
    max_length: int
    stop_words: List[str]


DEFAULT = LanguagePreset(
    max_length=128,
    stop_words=["\n\n"],
)


LanguagePresets = defaultdict(
    lambda: DEFAULT,
    [
        (
            Language.PYTHON,
            LanguagePreset(
                max_length=128,
                stop_words=["\n\n", "\ndef", "\n#", "\nimport", "\nfrom", "\nclass"],
            ),
        ),
        (
            Language.JAVASCRIPT,
            LanguagePreset(
                max_length=128,
                stop_words=["\n\n", "\nfunction", "\n//", "\nimport", "\nclass"],
            ),
        ),
        (
            Language.TYPESCRIPT,
            LanguagePreset(
                max_length=128,
                stop_words=[
                    "\n\n",
                    "\nfunction",
                    "\n//",
                    "\nimport",
                    "\nclass",
                    "\ninterface",
                    "\ntype",
                ],
            ),
        ),
    ],
)
