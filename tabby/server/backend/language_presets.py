from typing import List

from pydantic import BaseModel, Field

from ..models import Language


class LanguagePreset(BaseModel):
    max_length: int
    stop_words: List[str]


LanguagePresets = {
    Language.UNKNOWN: LanguagePreset(
        max_length=32,
        stop_words=["\n\n"],
    ),
    Language.PYTHON: LanguagePreset(
        max_length=128,
        stop_words=["\n\n", "\ndef", "\n#", "\nimport", "\nfrom", "\nclass"],
    ),
    Language.JAVASCRIPT: LanguagePreset(
        max_length=128, stop_words=["\n\n", "\nfunction", "\n//", "\nimport", "\nclass"]
    ),
}
