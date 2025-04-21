import { TextDocument } from "vscode-languageserver-textdocument";
import type { PostprocessFilter } from "./base";
import dedent from "dedent";
import { expect, AssertionError } from "chai";
import { v4 as uuid } from "uuid";
import { buildCompletionContext, CompletionContext, CompletionExtraContexts } from "../contexts";
import { CompletionResultItem } from "../solution";
import { splitLines } from "../../utils/string";

// `║` is the cursor position
export function documentContext(literals: TemplateStringsArray, ...placeholders: any[]): CompletionContext {
  const doc = dedent(literals, ...placeholders);
  const lines = splitLines(doc);
  const language = lines[0]?.trim() ?? "plaintext";
  const text = "\n" + lines.slice(1).join("");
  const textDocument = TextDocument.create(uuid(), language, 0, text.replace(/║/, ""));
  return buildCompletionContext(textDocument, textDocument.positionAt(text.indexOf("║")));
}

// `├` start of the inline completion to insert
// `┤` end of the inline completion to insert
// `┴` use for indent placeholder, should be placed at last line after `┤`

export function inline(literals: TemplateStringsArray, ...placeholders: any[]): string {
  const inline = dedent(literals, ...placeholders);
  return inline.slice(inline.indexOf("├") + 1, inline.lastIndexOf("┤"));
}

type TestCompletionItem = CompletionResultItem | string;

export async function assertFilterResult(
  filter: PostprocessFilter,
  context: CompletionContext & CompletionExtraContexts,
  input: TestCompletionItem,
  expected: TestCompletionItem,
) {
  const wrapTestCompletionItem = (testItem: TestCompletionItem): CompletionResultItem => {
    let item: CompletionResultItem;
    if (testItem instanceof CompletionResultItem) {
      item = testItem;
    } else {
      item = new CompletionResultItem(testItem);
    }
    return item;
  };
  const output = await filter(wrapTestCompletionItem(input), context, context);
  const expectedOutput = wrapTestCompletionItem(expected);
  expect(output.text).to.equal(expectedOutput.text);
}

export async function assertFilterResultNotEqual(
  filter: PostprocessFilter,
  context: CompletionContext,
  input: TestCompletionItem,
  expected: TestCompletionItem,
) {
  try {
    await assertFilterResult(filter, context, input, expected);
  } catch (error) {
    expect(error).to.be.instanceOf(AssertionError);
  }
}
