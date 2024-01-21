import { EditorState } from '@codemirror/state'

import { TCodeTag, TRange } from '@/app/files/components/source-code-browser'

/**
 * ranges encoded in tag are UTF-8 based, converting them into UTF-16 based range.
 */
function getUTF16NameRange(state: EditorState, tag: TCodeTag): TRange {
  const doc = state.doc
  const { span, utf16_column_range } = tag

  const line = doc.line(span.start.row + 1)
  const start = line.from + utf16_column_range.start
  const end = line.from + utf16_column_range.end
  return { start, end }
}

export { getUTF16NameRange }
