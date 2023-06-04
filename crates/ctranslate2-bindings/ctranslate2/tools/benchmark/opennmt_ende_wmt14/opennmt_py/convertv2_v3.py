import torch
import imp
import argparse
import pyonmttok
from onmt.constants import DefaultTokens
from onmt.inputters.inputter import vocabs_to_dict

# with the two module = imp.load_source() below
# we ghost the old torchtext.data.field and depercated
# onmt.inputters.text_dataset
# however this require some functions / classes to be
# monkey patched for loading the old field/vocab objects.


def _feature_tokenize():
    return 0


class RawField(object):
    def __init__(self):
        pass


class TextMultiField(RawField):
    def __init__(self):
        pass


class Field(RawField):
    def __init__(self):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v2model",
        type=str,
        required=True,
        help="Source OpenNMT-py v2.x model to be converted in v3.x",
    )
    parser.add_argument(
        "-v3model",
        type=str,
        required=True,
        help="Target model to be used by OpenNMT-py v3.x",
    )

    opt = parser.parse_args()
    print(opt)
    module1 = imp.load_source("torchtext.data.field", "convertv2_v3.py")
    module2 = imp.load_source("onmt.inputters.text_dataset", "convertv2_v3.py")
    checkpoint = torch.load(opt.v2model, map_location="cpu")
    vocabs = {}
    multifield = checkpoint["vocab"]["src"]
    multifields = multifield.fields
    _, fields = multifields[0]
    voc = fields.vocab.__dict__["itos"]
    src_vocab = pyonmttok.build_vocab_from_tokens(
        voc, maximum_size=0, minimum_frequency=1
    )
    src_vocab.default_id = src_vocab[DefaultTokens.UNK]
    vocabs["src"] = src_vocab
    print("Source vocab size is:", len(src_vocab))
    multifield = checkpoint["vocab"]["tgt"]
    multifields = multifield.fields
    _, fields = multifields[0]
    voc = fields.vocab.__dict__["itos"]
    tgt_vocab = pyonmttok.build_vocab_from_tokens(
        voc, maximum_size=0, minimum_frequency=1
    )
    tgt_vocab.default_id = src_vocab[DefaultTokens.UNK]
    vocabs["tgt"] = tgt_vocab
    print("Target vocab size is:", len(tgt_vocab))
    if hasattr(checkpoint["opt"], "data_task"):
        print("Model is type:", checkpoint["opt"].data_task)
        vocabs["data_task"] = checkpoint["opt"].data_task
    else:
        vocabs["data_task"] = "seq2seq"
    checkpoint["vocab"] = vocabs_to_dict(vocabs)

    checkpoint["opt"].__dict__["hidden_size"] = checkpoint["opt"].__dict__.pop(
        "rnn_size"
    )

    checkpoint["opt"].__dict__["add_qkvbias"] = True

    torch.save(checkpoint, opt.v3model)
