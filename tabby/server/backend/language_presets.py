from typing import List

from pydantic import BaseModel, Field


class LanguagePreset(BaseModel):
    max_length: int
    stop_words: List[str]


PythonPreset = LanguagePreset(
    max_length=128, stop_words=["\n\n", "\ndef", "\n#", "\nimport", "\nfrom", "\nclass"]
)

JavascriptPreset = LanguagePreset(
    max_length=128, stop_words=["\n\n", "\nfunction", "\n//", "\nimport", "\nclass"]
)

LanguagePresets = {"python": PythonPreset, "javascript": JavascriptPreset}
