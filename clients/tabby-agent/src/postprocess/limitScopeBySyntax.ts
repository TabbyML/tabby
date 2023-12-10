import type TreeSitterParser from "web-tree-sitter";
import { getParser, languagesConfigs } from "../syntax/parser";
import { typeList } from "../syntax/typeList";
import { CompletionContext } from "../CompletionContext";
import { PostprocessFilter, logger } from "./base";

export const supportedLanguages = Object.keys(languagesConfigs);

function findLineBegin(text: string, position: number): number {
  let lastNonBlankCharPos = position - 1;
  while (lastNonBlankCharPos >= 0 && text[lastNonBlankCharPos]?.match(/\s/)) {
    lastNonBlankCharPos--;
  }
  if (lastNonBlankCharPos < 0) {
    return 0;
  }
  const lineBegin = text.lastIndexOf("\n", lastNonBlankCharPos);
  if (lineBegin < 0) {
    return 0;
  }
  const line = text.slice(lineBegin + 1, position);
  const indentation = line.search(/\S/);
  return lineBegin + 1 + indentation;
}

function findLineEnd(text: string, position: number): number {
  let firstNonBlankCharPos = position;
  while (firstNonBlankCharPos < text.length && text[firstNonBlankCharPos]?.match(/\s/)) {
    firstNonBlankCharPos++;
  }
  if (firstNonBlankCharPos >= text.length) {
    return text.length;
  }
  const lineEnd = text.indexOf("\n", firstNonBlankCharPos);
  if (lineEnd < 0) {
    return text.length;
  }
  return lineEnd;
}

function findScope(node: TreeSitterParser.SyntaxNode, typeList: string[][]): TreeSitterParser.SyntaxNode {
  for (const types of typeList) {
    let scope: TreeSitterParser.SyntaxNode | null = node;
    while (scope) {
      if (types.includes(scope.type)) {
        return scope;
      }
      scope = scope.parent;
    }
  }
  return node;
}

export function limitScopeBySyntax(): PostprocessFilter {
  return async (input: string, context: CompletionContext) => {
    const { position, text, language, prefix, suffix } = context;
    if (!supportedLanguages.includes(language)) {
      return input;
    }
    const languageConfig = languagesConfigs[language]!;
    const parser = await getParser(languageConfig);

    const updatedText = prefix + input + suffix;
    const updatedTree = parser.parse(updatedText);
    const lineBegin = findLineBegin(updatedText, position);
    const lineEnd = findLineEnd(updatedText, position);
    const scope = findScope(
      updatedTree.rootNode.namedDescendantForIndex(lineBegin, lineEnd),
      typeList[languageConfig] ?? [],
    );

    if (scope.endIndex < position + input.length) {
      logger.debug(
        {
          languageConfig,
          text,
          updatedText,
          position,
          lineBegin,
          lineEnd,
          scope: { type: scope.type, start: scope.startIndex, end: scope.endIndex },
        },
        "Remove content out of syntax scope",
      );
      return input.slice(0, scope.endIndex - position);
    }
    return input;
  };
}
