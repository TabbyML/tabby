import type { PostprocessFilter } from "./base";
import dedent from "dedent";
import { expect, AssertionError } from "chai";
import { v4 as uuid } from "uuid";
import { CompletionContext } from "../contexts";
import { CompletionItem } from "../solution";

// `║` is the cursor position
export function documentContext(literals: TemplateStringsArray, ...placeholders: any[]): CompletionContext {
  const doc = dedent(literals, ...placeholders);
  return new CompletionContext({
    filepath: uuid(),
    language: "",
    text: doc.replace(/║/, ""),
    position: doc.indexOf("║"),
  });
}

// `├` start of the inline completion to insert
// `┤` end of the inline completion to insert
// `┴` use for indent placeholder, should be placed at last line after `┤`

export function inline(literals: TemplateStringsArray, ...placeholders: any[]): string {
  const inline = dedent(literals, ...placeholders);
  return inline.slice(inline.indexOf("├") + 1, inline.lastIndexOf("┤"));
}

type TestCompletionItem = CompletionItem | string | { text: string; replacePrefix?: string; replaceSuffix?: string };

export async function assertFilterResult(
  filter: PostprocessFilter,
  context: CompletionContext,
  input: TestCompletionItem,
  expected: TestCompletionItem,
) {
  const wrapTestCompletionItem = (testItem: TestCompletionItem): CompletionItem => {
    let item: CompletionItem;
    if (testItem instanceof CompletionItem) {
      item = testItem;
    } else if (typeof testItem === "string") {
      item = new CompletionItem(context, testItem);
    } else {
      item = new CompletionItem(context, testItem.text, testItem.replacePrefix, testItem.replaceSuffix);
    }
    return item;
  };
  const output = await filter(wrapTestCompletionItem(input));
  const expectedOutput = wrapTestCompletionItem(expected);
  expect(output.text).to.equal(expectedOutput.text);
  expect(output.replacePrefix).to.equal(expectedOutput.replacePrefix);
  expect(output.replaceSuffix).to.equal(expectedOutput.replaceSuffix);
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
