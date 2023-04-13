import os
import re

import meilisearch
from loguru import logger

from .language_presets import LanguagePreset

FLAGS_enable_meilisearch = os.environ.get("FLAGS_enable_meilisearch", None)


class PromptRewriter:
    def __init__(self, meili_addr: str = "http://localhost:8084"):
        self.meili_client = meilisearch.Client(meili_addr)

    def create_query(self, preset: LanguagePreset, prompt: str):
        # Remove all punctuations and create tokens.
        tokens = re.sub(r"[^\w\s]", " ", prompt.lower()).split()

        # Remove short tokens.
        tokens = [x for x in tokens if len(x) >= 3]

        # Remove tokens in language reserved_keywords.
        tokens = set([x for x in tokens if x not in preset.reserved_keywords])

        if len(tokens) > 3:
            return " ".join(tokens)
        else:
            raise PromptRewriteFailed("Too few tokens extracted from prompt")

    def rewrite(self, preset: LanguagePreset, prompt: str) -> str:
        if preset.reserved_keywords is None:
            raise PromptRewriteFailed("Rewrite requires language keywords list")

        index = self.meili_client.index("dataset")
        query = self.create_query(preset, prompt)
        logger.debug("query: {}", query)
        search_results = index.search(
            query,
            {
                "limit": 3,
                "attributesToCrop": ["content"],
                "cropLength": 32,
                "cropMarker": "",
                "attributesToRetrieve": ["content"],
            },
        )

        if len(search_results["hits"]) == 0:
            raise PromptRewriteFailed("No related snippets")

        def make_snippet(i, content):
            content = content["_formatted"]["content"]
            return f"== snippet {i+1} ==\n{content}"

        snippets = "\n".join(
            [make_snippet(i, x) for i, x in enumerate(search_results["hits"])]
        )
        prompt = f"""Given following relevant code snippet, generate code completion based on context.
{snippets}
== context ==
{prompt}"""
        logger.debug("prompt: {}", prompt)
        return prompt

    def __call__(self, preset: LanguagePreset, prompt: str) -> str:
        try:
            return self.rewrite(preset, prompt)
        except PromptRewriteFailed:
            return prompt


class PromptRewriteFailed(Exception):
    pass
