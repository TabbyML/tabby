import os
import glob
import json

from datasets import Dataset
from transformers import HfArgumentParser

import metrics
from args import PreprocessProjectArgs


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


def dataset_iter(files):
    def gen():
        mappings = read_extension_to_language_mappings()
        for x in files:
            _, extname = os.path.splitext(x)

            with open(x) as f:
                content = f.read()

            yield dict(
                language=mappings[extname],
                content=content,
                **metrics.compute(content),
            )

    return gen


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

    ds = Dataset.from_generator(dataset_iter(files))
    ds.save_to_disk(args.output_dir)
