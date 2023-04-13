from typing import List, Optional, Set

from pydantic import BaseModel, Field

from ..models import Language


class LanguagePreset(BaseModel):
    max_length: int
    stop_words: List[str]
    reserved_keywords: Optional[Set]


LanguagePresets = {
    Language.UNKNOWN: LanguagePreset(
        max_length=128,
        stop_words=["\n\n"],
    ),
    Language.PYTHON: LanguagePreset(
        max_length=128,
        stop_words=["\n\n", "\ndef", "\n#", "\nimport", "\nfrom", "\nclass"],
        reserved_keywords=set(
            [
                "False",
                "class",
                "from",
                "or",
                "None",
                "continue",
                "global",
                "pass",
                "True",
                "def",
                "if",
                "raise",
                "and",
                "del",
                "import",
                "return",
                "as",
                "elif",
                "in",
                "try",
                "assert",
                "else",
                "is",
                "while",
                "async",
                "except",
                "lambda",
                "with",
                "await",
                "finally",
                "nonlocal",
                "yield",
                "break",
                "for",
                "not",
            ]
        ),
    ),
    Language.JAVASCRIPT: LanguagePreset(
        max_length=128, stop_words=["\n\n", "\nfunction", "\n//", "\nimport", "\nclass"]
    ),
    Language.TYPESCRIPT: LanguagePreset(
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
}
