from dataclasses import dataclass, field
from typing import Optional


@dataclass(kw_only=True)
class FilterArgs:
    line_max: Optional[int] = field(
        default=1000,
        metadata={"help": "Max line length allowed"},
    )
    line_mean: Optional[int] = field(
        default=100,
        metadata={"help": "Mean line length allowed"},
    )
    alpha_frac: Optional[float] = field(
        default=0.25,
        metadata={"help": "Minimum fraction of alphanumeric characters allowed."},
    )


@dataclass
class PreprocessProjectArgs(FilterArgs):
    # add arguments in the following format
    project_dir: Optional[str] = field(
        metadata={"help": "Project directory."},
    )

    output_dir: Optional[str] = field(
        metadata={"help": "Output save path directory."},
    )
