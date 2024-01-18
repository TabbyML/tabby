import {
  defaultHighlightStyle,
  foldGutter,
  syntaxHighlighting
} from '@codemirror/language'
import { highlightSelectionMatches } from '@codemirror/search'
import { EditorState, Extension } from '@codemirror/state'
import {
  EditorView,
  highlightSpecialChars,
  lineNumbers,
  rectangularSelection
} from '@codemirror/view'

const basicTheme = EditorView.baseTheme({
  '.cm-focused': {
    outline: 'none !important'
  }
})

export const basicSetup: Extension = (() => [
  basicTheme,
  lineNumbers(),
  highlightSpecialChars(),
  highlightSelectionMatches(),
  EditorState.allowMultipleSelections.of(true),
  syntaxHighlighting(defaultHighlightStyle, {
    fallback: true
  }),
  rectangularSelection(),
  foldGutter({
    markerDOM(open) {
      const dom = document.createElement('div')
      dom.style.cursor = 'pointer'
      if (open) {
        dom.innerHTML =
          '<svg aria-hidden="true" focusable="false" role="img" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" style="display: inline-block; user-select: none; vertical-align: text-bottom; overflow: visible;"><path d="M12.78 5.22a.749.749 0 0 1 0 1.06l-4.25 4.25a.749.749 0 0 1-1.06 0L3.22 6.28a.749.749 0 1 1 1.06-1.06L8 8.939l3.72-3.719a.749.749 0 0 1 1.06 0Z"></path></svg>'
      } else {
        dom.innerHTML =
          '<svg aria-hidden="true" focusable="false" role="img" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" style="display: inline-block; user-select: none; vertical-align: text-bottom; overflow: visible;"><path d="M6.22 3.22a.75.75 0 0 1 1.06 0l4.25 4.25a.75.75 0 0 1 0 1.06l-4.25 4.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042L9.94 8 6.22 4.28a.75.75 0 0 1 0-1.06Z"></path></svg>'
      }

      return dom
    }
  })
])()
