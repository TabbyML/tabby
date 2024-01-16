import { defaultHighlightStyle, syntaxHighlighting } from '@codemirror/language'
import { highlightSelectionMatches } from '@codemirror/search'
import { EditorState, Extension } from '@codemirror/state'
import {
  EditorView,
  highlightActiveLine,
  highlightActiveLineGutter,
  highlightSpecialChars,
  lineNumbers,
  rectangularSelection
} from '@codemirror/view'

const basicTheme = EditorView.baseTheme({
  '.ͼ1.cm-focused': {
    outline: 'none !important'
  }
})

export const basicSetup: Extension = (() => [
  basicTheme,
  lineNumbers(),
  highlightActiveLineGutter(),
  highlightSpecialChars(),
  highlightActiveLine(),
  highlightSelectionMatches(),
  EditorState.allowMultipleSelections.of(true),
  syntaxHighlighting(defaultHighlightStyle, {
    fallback: true
  }),
  rectangularSelection()
])()
