import tarfile
from dataclasses import dataclass, field
from typing import Optional

import requests
from transformers import HfArgumentParser


@dataclass(kw_only=True)
class DownloaderArgs:
    url: str = field(metadata={"help": "URL to source code tar.gz file"})
    output_dir: str = field(metadata={"help": "Output save path directory"})


def parse_args():
    parser = HfArgumentParser(DownloaderArgs)
    return parser.parse_args()


def download_and_untar(url, output_dir):
    response = requests.get(url, stream=True)
    mode = "r"
    if url.endswith(".gz"):
        mode += "|gz"
    elif url.endswith(".xz"):
        mode += "|xz"
    elif url.endswith(".bz2"):
        mode += "|bz2"
    file = tarfile.open(fileobj=response.raw, mode=mode)
    file.extractall(output_dir)


if __name__ == "__main__":
    args = parse_args()
    download_and_untar(args.url, args.output_dir)
