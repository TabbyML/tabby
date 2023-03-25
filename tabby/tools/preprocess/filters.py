from .args import FilterArgs


def basic_filters(args: FilterArgs):
    def fn(example):
        """Filter files based on line length and % alphanumeric characters"""
        if example["max_line_length"] > args.line_max:
            return False
        elif example["avg_line_length"] > args.line_mean:
            return False
        elif example["alphanum_fraction"] < args.alpha_frac:
            return False
        return True

    return fn
