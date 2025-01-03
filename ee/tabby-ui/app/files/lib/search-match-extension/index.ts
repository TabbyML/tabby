import { Range } from '@codemirror/state'
import {
  Decoration,
  DecorationSet,
  EditorView,
  ViewPlugin,
  ViewUpdate
} from '@codemirror/view'

import './style.css'

const searchMark = Decoration.mark({ class: 'search-match-mark' })
const searchMarkTheme = EditorView.theme({
  '.search-match-mark': {
    backgroundColor: 'hsl(var(--mark-bg))'
  }
})

function getMatches(
  view: EditorView,
  matches: { bytesStart: number; bytesEnd: number; lineNumber: number }[] = []
) {
  const doc = view.state.doc
  const len = doc.length
  if (!len) return Decoration.none

  let widgets: Range<Decoration>[] = []
  for (const match of matches) {
    const line = doc.line(match.lineNumber)
    const startPos = line.from
    const range = {
      start: startPos + match.bytesStart,
      end: startPos + match.bytesEnd
    }
    try {
      if (range && range.start <= len && range.end <= len) {
        widgets.push(searchMark.range(range.start, range.end))
      }
    } catch (e) {}
  }

  if (!widgets.length) return Decoration.none
  return Decoration.set(widgets)
}

const searchMatchExtension = (
  matches: { bytesStart: number; bytesEnd: number; lineNumber: number }[]
) => {
  const extension = ViewPlugin.fromClass(
    class {
      marks: DecorationSet
      constructor(view: EditorView) {
        this.marks = getMatches(view, matches)
      }
      update(update: ViewUpdate) {
        if (update.docChanged || update.viewportChanged) {
          this.marks = getMatches(update.view, matches)
        }
      }
    },
    {
      decorations: instance => {
        return instance.marks
      }
    }
  )

  return [extension, searchMarkTheme]
}

export { searchMatchExtension, searchMarkTheme }
