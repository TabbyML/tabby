import { EditorState } from '@codemirror/state'

import { TCodeTag } from '@/app/files/components/source-code-browser'

/**
 * resolve the range offset caused by emoji encoding
 */
function getRangeOffset(state: EditorState, tag: TCodeTag): number {
  if (!tag) return 0

  const { utf16_column_range, name_range } = tag

  try {
    const line = state.doc.lineAt(name_range.start)
    const column = name_range.start - line.from
    return  utf16_column_range.start - column
  } catch (e) {
    return 0
  }
}

export { getRangeOffset }
