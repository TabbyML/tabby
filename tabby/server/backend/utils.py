import random
import string


def random_completion_id():
    return "cmpl-" + "".join(
        random.choice(string.ascii_letters + string.digits) for _ in range(29)
    )


def trim_with_stopwords(output: str, stopwords: list) -> str:
    for w in sorted(stopwords, key=len, reverse=True):
        if output.endswith(w):
            output = output[: -len(w)]
            break
    return output
