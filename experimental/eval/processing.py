from typing import Iterator

import glob
import json
from dataclasses import dataclass


@dataclass
class Item:
    git_url: str
    filepath: str
    language: str

    name: str
    body: str
    prefix: str
    suffix: str


def iter_items(doc) -> Iterator[Item]:
    if doc["max_line_length"] > 500:
        return

    if doc["avg_line_length"] < 10 or doc["avg_line_length"] > 200:
        return

    if doc["alphanum_fraction"] < 0.25:
        return

    for tag in doc["tags"]:
        content = doc["content"]
        name = get_content(content, tag["name_range"])
        body = get_content(content, tag["range"])

        prefix = get_prefix(content, tag["range"]["start"])
        suffix = get_suffix(content, tag["range"]["end"])

        yield Item(
            name=name,
            body=body,
            prefix=prefix,
            suffix=suffix,
            git_url=doc["git_url"],
            filepath=doc["filepath"],
            language=doc["language"],
        )


def iter_docs(filepattern: str):
    for filepath in glob.glob(filepattern):
        with open(filepath) as f:
            for line in f:
                yield json.loads(line)


def get_content(content: str, range: dict):
    return content[range["start"] : range["end"]]


def get_prefix(content: str, start: int, max=20):
    num_lines = 0
    prefix_start = 0
    for prefix_start in range(start - 1, 0, -1):
        if content[prefix_start] == "\n":
            num_lines += 1

        if num_lines == max:
            break

    return content[prefix_start + 1 : start]


def get_suffix(content: str, end: int, max=20):
    num_lines = 0
    suffix_end = end
    for suffix_end in range(end, len(content)):
        if content[suffix_end] == "\n":
            num_lines += 1

        if num_lines == max:
            break

    return content[end : suffix_end - 1]


def items_from_filepattern(filepattern: str):
    for doc in iter_docs(filepattern):
        yield from iter_items(doc)
