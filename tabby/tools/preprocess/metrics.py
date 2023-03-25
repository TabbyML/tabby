def max_line_length(content):
    return max([0] + [len(x) for x in content.splitlines()])


def avg_line_length(content):
    lines = [len(x) for x in content.splitlines()]
    total = sum(lines)
    if len(lines) != 0:
        return total / len(lines)
    else:
        return 0


def alphanum_fraction(content):
    alphanum = [x for x in content if x.isalpha() or x.isnumeric()]
    if len(content) != 0:
        return len(alphanum) / len(content)
    else:
        return 0


def compute(content):
    return dict(
        max_line_length=max_line_length(content),
        avg_line_length=avg_line_length(content),
        alphanum_fraction=alphanum_fraction(content),
    )
