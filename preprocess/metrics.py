def max_line_length(content):
    return max([len(x) for x in content.splitlines()])


def avg_line_length(content):
    lines = [len(x) for x in content.splitlines()]
    total = sum(lines)
    return total / len(lines)


def alphanum_fraction(content):
    alphanum = [x for x in content if x.isalpha() or x.isnumeric()]
    return len(alphanum) / len(content)


def compute(content):
    return dict(
        max_line_length=max_line_length(content),
        avg_line_length=avg_line_length(content),
        alphanum_fraction=alphanum_fraction(content),
    )
