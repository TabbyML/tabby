import os
import pathlib
from dataclasses import dataclass, field

import toml
from git import Repo
from transformers import HfArgumentParser


@dataclass
class Arguments:
    data_dir: str = field(metadata={"help": "Base dir for repositories"})
    config_file: str = field(metadata={"help": "Configuration file for tabby updater"})


def parse_args():
    parser = HfArgumentParser(Arguments)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    config = toml.load(args.config_file)
    repositories = config["repositories"]

    for name, config in repositories.items():
        path = pathlib.Path(args.data_dir, name)
        if path.is_dir():
            repo = Repo(path)
        else:
            Repo.clone_from(config["url"], path.absolute(), depth=1)

    os.system(f"gitup {args.data_dir}")
