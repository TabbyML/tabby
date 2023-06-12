import torch
from datasets import Dataset, load_from_disk


class ConstantLengthDataset:
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.
        Args:
            tokenizer (Tokenizer): The processor used for proccessing the data.
            dataset (dataset.Dataset): Dataset with text files.
            infinite (bool): If True the iterator is reset after dataset reaches end else stops.
            seq_length (int): Length of token sequences to return.
            num_of_sequences (int): Number of token sequences to keep in buffer.
            chars_per_token (int): Number of characters per token used to estimate number of tokens in text buffer.
    """

    def __init__(
        self,
        tokenizer,
        dataset,
        infinite=False,
        seq_length=1024,
        num_of_sequences=1024,
        chars_per_token=3.6,
        content_field="content",
    ):
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.eos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.infinite = infinite
        self.current_size = 0
        self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
        self.content_field = content_field

    def __call__(self):
        def gen():
            for x in self:
                yield x

        return gen()

    def __iter__(self):
        for buffer in self._read_dataset_into_buffer():
            yield from self._tokenize(buffer)

    def _tokenize(self, buffer):
        tokenized_inputs = self.tokenizer(buffer, truncation=False)["input_ids"]

        all_token_ids = []
        for tokenized_input in tokenized_inputs:
            all_token_ids.extend(tokenized_input + [self.concat_token_id])

        for i in range(0, len(all_token_ids), self.seq_length):
            input_ids = all_token_ids[i : i + self.seq_length]

            if len(input_ids) < self.seq_length:
                input_ids = all_token_ids[-self.seq_length :]

            if len(input_ids) == self.seq_length:
                self.current_size += 1
                yield dict(input_ids=input_ids, labels=input_ids)

    def _read_dataset_into_buffer(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    buffer.append(next(iterator)[self.content_field])
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                    else:
                        more_examples = False
                        break
            yield buffer


def load_dataset(tokenizer, filepath, **kwargs):
    ds = load_from_disk(filepath)
    ds = Dataset.from_generator(ConstantLengthDataset(tokenizer, ds, **kwargs))
    return ds
