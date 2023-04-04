from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PreprocessProjectArgs:
    # add arguments in the following format
    project_dir: Optional[str] = field(
        metadata={"help": "Project directory."},
    )

    output_dir: Optional[str] = field(
        metadata={"help": "Output save path directory."},
    )
