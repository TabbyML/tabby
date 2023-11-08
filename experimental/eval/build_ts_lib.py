#!/usr/bin/env python
# coding=utf-8
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

from tree_sitter import Language

def build_language_lib():
    for lang in ["java", "python", "typescript", "csharp"]:
        ts_lang = "c-sharp" if lang == "csharp" else lang
        if lang == "typescript":
            git_dir = f"ts_package/tree-sitter-{ts_lang}/{lang}"
        else:
            git_dir = f"ts_package/tree-sitter-{ts_lang}"
        Language.build_library(f'build/{lang}-lang-parser.so', [git_dir])


if __name__ == "__main__":
    build_language_lib()
