import dedent from "dedent";
import type { PostprocessContext } from "./filter";

// `║` is the cursor position

export function documentContext(strings): PostprocessContext {
  const doc = dedent(strings);
  return {
    filepath: null,
    language: null,
    text: doc.replace(/║/, ""),
    position: doc.indexOf("║"),
    maxPrefixLines: 20,
    maxSuffixLines: 20,
  };
}

// `├` start of the inline completion to insert
// `┤` end of the inline completion to insert
// `┴` use for indent placeholder, should be placed at last line after `┤`

export function inline(strings): string {
  const inline = dedent(strings);
  return inline.slice(inline.indexOf("├") + 1, inline.lastIndexOf("┤"));
}
