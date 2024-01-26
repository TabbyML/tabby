import { EditorState } from '@codemirror/state'

import type { TCodeTag, TRange } from '@/lib/types'

/**
 * ranges encoded in tag are UTF-8 based, converting them into UTF-16 based range.
 */
function getUTF16NameRange(state: EditorState, tag: TCodeTag): TRange | null {
  const doc = state.doc
  const { span, utf16_column_range } = tag
  try {
    const line = doc.line(span.start.row + 1)
    const start = line.from + utf16_column_range.start
    const end = line.from + utf16_column_range.end
    return { start, end }
  } catch (e) {
    return null
  }
}

export { getUTF16NameRange }
