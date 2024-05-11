import { defaultHighlightStyle, syntaxHighlighting } from '@codemirror/language'
import { highlightSelectionMatches } from '@codemirror/search'
import { EditorState, Extension } from '@codemirror/state'
import { highlightSpecialChars, rectangularSelection } from '@codemirror/view'

export const basicSetup: Extension = (() => [
  highlightSpecialChars(),
  highlightSelectionMatches(),
  EditorState.allowMultipleSelections.of(true),
  syntaxHighlighting(defaultHighlightStyle, {
    fallback: true
  }),
  rectangularSelection()
])()
