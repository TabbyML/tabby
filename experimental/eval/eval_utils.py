# Copyright Amazon.com, Inc. or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import ast
import re
from functools import lru_cache
from typing import List

import timeout_decorator
import torch
from fuzzywuzzy import fuzz
from nltk.tokenize import RegexpTokenizer
from sacrebleu.tokenizers.tokenizer_intl import TokenizerV14International

from keywords.keywordlist import get_language_keywords

IDENTIFIER_REGEX = re.compile('[_a-zA-Z][_a-zA-Z0-9]*')
REGEX_TEXT = ("(?<=[a-z0-9])(?=[A-Z])|"
              "(?<=[A-Z0-9])(?=[A-Z][a-z])|"
              "(?<=[0-9])(?=[a-zA-Z])|"
              "(?<=[A-Za-z])(?=[0-9])|"
              "(?<=[@$.'\"])(?=[a-zA-Z0-9])|"
              "(?<=[a-zA-Z0-9])(?=[@$.'\"])|"
              "_|\\s+")
string_pattern = r'"([^"\\]*(\\.[^"\\]*)*)"|\'([^\'\\]*(\\.[^\'\\]*)*)\''

SPLIT_REGEX = re.compile(REGEX_TEXT)

str_tokenizer = TokenizerV14International()
code_tokenizer = RegexpTokenizer(r'\w+')


def cal_edit_sim(references, hypotheses):
    total = len(references)
    edit_sim = 0.0
    for pred, gt in zip(hypotheses, references):
        pred = pred.strip()
        gt = gt.strip()
        edit_sim += fuzz.ratio(pred, gt)
    return edit_sim / total


@lru_cache(maxsize=5000)
def split_identifier_into_parts(identifier: str) -> List[str]:
    """
    Split a single identifier into parts on snake_case and camelCase
    """
    identifier_parts = list(s for s in SPLIT_REGEX.split(identifier) if len(s) > 0)

    if len(identifier_parts) == 0:
        return [identifier]
    if "_" in identifier:  # We consider "_" as part of identifier and add it back in between each semantic part
        # if snake_case, we only split identifiers based on "_", ignore the mixed camelCase or other special symbols
        # this helps us avoid splitting identifiers like "get_2d_array" into ["get", "2", "d", "array"]
        # also avoid many other corner cases
        identifier_parts = identifier.split("_")
        tmp = [identifier_parts[0]]
        for i in identifier_parts[1:]:
            tmp.append("_")
            tmp.append(i)
        identifier_parts = tmp

    return identifier_parts


def is_identifier(token, lang=None):
    return True if IDENTIFIER_REGEX.match(token) \
                   and (lang is None or token not in get_language_keywords(lang)) \
        else False


def extract_identifiers(source_code, lang):
    # the main idea is to remove String from a source code
    # then, tokenize the code to get all words and match with identifier regular expression
    # check if it is a language specific keyword, it not, then it is an identifier
    source_code_without_strings = re.sub(string_pattern, '', source_code)
    _ids = [t for t in code_tokenizer.tokenize(source_code_without_strings) if is_identifier(t, lang)]
    return _ids


def tokenize_string(input_str):
    return str_tokenizer(input_str)


def get_bracket_lang_statement(completion):
    end_idx = None
    for i in range(len(completion)):
        if completion[i] in [";", "}", "{"]:
            end_idx = i
            break
    return completion[:end_idx + 1] if end_idx else completion


@timeout_decorator.timeout(5)
def get_ast(parser, code):
    assert isinstance(code, str) or isinstance(code, bytes)
    if isinstance(code, str):
        code = bytes(code, "utf8")
    try:
        tree = parser.parse(code)
        return tree
    except Exception as e:
        return None


def remove_comments(code):
    code = re.sub(r'#.*', '', code)
    code = re.sub(r'//.*', '', code)
    return code


def is_parse_valid(parser, code):
    def syntax_error(node):
        if node.type == "ERROR":
            return True
        try:
            for child in node.children:
                if syntax_error(child):
                    return True
        except RecursionError as err:
            return True

        return False

    tree = get_ast(parser, code)
    if tree is not None:
        return not syntax_error(tree.root_node)
    return False


def is_code_parseable(code):
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def get_python_one_statement(prompt, completion, parser):
    for i in range(len(completion)):
        code = prompt + completion[:i + 1]
        if not is_parse_valid(parser, code):
            continue
        if completion[i + 1] == "\n":
            return completion[:i + 1].rstrip()

    return completion


def postprocess_code_lines(prompt, completion, parser, lang):
    try:
        if lang in ["java", "csharp", "typescript"]:
            return get_bracket_lang_statement(completion)
        elif lang == "python":
            return get_python_one_statement(prompt, completion, parser)
    except Exception as e:
        return completion


def compute_mean_logp(scores, sequences, pad_token_id):
    assert scores.shape[0] == sequences.shape[0]
    assert scores.shape[1] == sequences.shape[1]
    with torch.no_grad():
        logp_vocab = torch.nn.functional.log_softmax(scores, dim=-1)
        indices = torch.unsqueeze(sequences, dim=-1)
        logp = torch.gather(logp_vocab, dim=-1, index=indices).squeeze(-1)
        sum_logp = torch.cumsum(logp, dim=1)  # batch_size, seq_len
        denom = torch.arange(1, sum_logp.shape[1] + 1).reshape(1, -1).to(device=sum_logp.device)  # 1, seq_len
        mean_logp = (sum_logp / denom).tolist()  # batch_size, seq_len
        sequence_lengths = (sequences != pad_token_id).sum(1).tolist()  # batch_size
        mean_logp = [mean_logp[idx][l - 1] for idx, l in enumerate(sequence_lengths)]
    return mean_logp
