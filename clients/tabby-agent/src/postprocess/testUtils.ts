import dedent from "dedent";

// `║` is the cursor position

export function documentContext(strings) {
  const doc = dedent(strings);
  return {
    filepath: null,
    text: doc.replace(/║/, ""),
    position: doc.indexOf("║"),
  };
}

// `├` start of the inline completion to insert
// `┤` end of the inline completion to insert
// `┴` use for indent placeholder, should be placed at last line after `┤`

export function inline(strings) {
  const inline = dedent(strings);
  return inline.slice(inline.indexOf("├") + 1, inline.lastIndexOf("┤"));
}
