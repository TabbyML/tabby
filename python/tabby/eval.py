from typing import Iterator, Optional

import glob
import json
from dataclasses import dataclass, field
from transformers import HfArgumentParser


@dataclass
class Arguments:
    filepattern: str = field(metadata={"help": "filepattern for tabby dataset"})


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


def parse_args() -> Arguments:
    parser = HfArgumentParser(Arguments)
    return parser.parse_args()


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


if __name__ == "__main__":
    args = parse_args()
    for doc in iter_docs(args.filepattern):
        for item in iter_items(doc):
            pass
