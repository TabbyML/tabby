import dedent from "dedent";
import { v4 as uuid } from "uuid";
import { CompletionContext } from "../CompletionContext";

// `║` is the cursor position
export function documentContext(strings): CompletionContext {
  const doc = dedent(strings);
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

export function inline(strings): string {
  const inline = dedent(strings);
  return inline.slice(inline.indexOf("├") + 1, inline.lastIndexOf("┤"));
}
