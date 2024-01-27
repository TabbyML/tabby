import dedent from "dedent";
import { v4 as uuid } from "uuid";
import { CompletionContext } from "../CompletionContext";

// `║` is the cursor position
export function documentContext(literals: TemplateStringsArray, ...placeholders: any[]): CompletionContext {
  const doc = dedent(literals, ...placeholders);
  return new CompletionContext({
    path: null,
    filepath: uuid(),
    language: "",
    text: doc.replace(/║/, ""),
    position: doc.indexOf("║"),
    maxPrefixLines: 20,
    maxSuffixLines: 20,
  });
}

// `├` start of the inline completion to insert
// `┤` end of the inline completion to insert
// `┴` use for indent placeholder, should be placed at last line after `┤`

export function inline(literals: TemplateStringsArray, ...placeholders: any[]): string {
  const inline = dedent(literals, ...placeholders);
  return inline.slice(inline.indexOf("├") + 1, inline.lastIndexOf("┤"));
}
