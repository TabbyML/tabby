import base64
import glob
import json
import os

import pandas as pd
from datasets import Dataset
from transformers import HfArgumentParser

from . import metrics
from .args import PreprocessProjectArgs


def parse_args():
    parser = HfArgumentParser(PreprocessProjectArgs)
    return parser.parse_args()


def read_languages_to_file_extensions():
    path = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(path, "programming-languages-to-file-extensions.json")
    with open(path) as f:
        return json.load(f)


def read_valid_extensions():
    content = read_languages_to_file_extensions()
    extensions = []
    for k, exts in content.items():
        extensions += exts
    return set(extensions)


def read_extension_to_language_mappings():
    content = read_languages_to_file_extensions()
    mappings = dict()
    for k, exts in content.items():
        for x in exts:
            mappings[x] = k
    return mappings


def dataset_iter(project_dir, files):
    def gen():
        mappings = read_extension_to_language_mappings()
        for x in files:
            _, extname = os.path.splitext(x)

            with open(x) as f:
                try:
                    content = f.read()
                except UnicodeDecodeError:
                    print("Cannot decode unicode", x)
                    continue

            segments = x.removeprefix(project_dir).split(os.sep)
            project = segments[1]
            file = os.path.join(*segments[2:])
            yield dict(
                id=to_id(project, file),
                project=project,
                file=file,
                language=mappings[extname],
                content=content,
                **metrics.compute(content),
            )

    return gen


def count_by_language(dataset):
    key = "language"
    df = (
        pd.DataFrame(dataset[key], columns=[key])
        .groupby([key])
        .size()
        .to_frame("count")
    )
    return df


def to_id(*args):
    token = ":".join(args)
    return base64.urlsafe_b64encode(token.encode("utf-8")).decode("utf-8").rstrip("=")


if __name__ == "__main__":
    valid_extensions = read_valid_extensions()

    def is_valid_file(x):
        if not os.path.isfile(x):
            return False

        _, extname = os.path.splitext(x)
        if not extname in valid_extensions:
            return False

        return True

    args = parse_args()
    files = list(
        filter(is_valid_file, glob.glob(args.project_dir + "/**/*", recursive=True))
    )

    ds = Dataset.from_generator(dataset_iter(os.path.abspath(args.project_dir), files))
    ds.save_to_disk(args.output_dir)
    ds.to_json(os.path.join(args.output_dir, "dumps.json"))

    print("\n## Summary")
    print("Number of source files", len(ds))
    print("Number of source files by languages", count_by_language(ds).to_json())
