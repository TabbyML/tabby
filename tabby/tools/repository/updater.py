import os
import pathlib
import shutil
from dataclasses import dataclass, field

import toml
from git import Repo
from loguru import logger
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
    repositories = config["projects"]

    for x in pathlib.Path(args.data_dir).glob("*"):
        if x.is_dir() and x.name not in repositories:
            print("Remove unused dir:", x)
            shutil.rmtree(str(x))
        elif x.is_file():
            print("Remove unused file:", x)
            x.unlink()

    for name, config in repositories.items():
        path = pathlib.Path(args.data_dir, name)
        if path.is_dir():
            repo = Repo(path)
        else:
            Repo.clone_from(config["git_url"], path.absolute(), depth=1)

    logger.info("Number of projects {}", len(repositories))
    os.system(f"gitup {args.data_dir}")
