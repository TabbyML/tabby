import random
import string


def random_completion_id():
    return "cmpl-" + "".join(
        random.choice(string.ascii_letters + string.digits) for _ in range(29)
    )


def trim_with_stop_words(output: str, stopwords: list) -> str:
    for w in sorted(stopwords, key=len, reverse=True):
        index = output.find(w)
        if index != -1:
            output = output[:index]
    return output
