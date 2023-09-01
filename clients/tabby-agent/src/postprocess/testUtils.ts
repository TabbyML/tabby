import dedent from "dedent";
import { buildContext, PostprocessContext } from "./base";

// `║` is the cursor position

export function documentContext(strings): PostprocessContext {
  const doc = dedent(strings);
  return buildContext({
    filepath: null,
    language: null,
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
