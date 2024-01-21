import { Range } from '@codemirror/state'
import {
  Decoration,
  DecorationSet,
  EditorView,
  ViewPlugin,
  ViewUpdate
} from '@codemirror/view'

import { TCodeTag } from '@/app/files/components/source-code-browser'

const tagMark = Decoration.mark({ class: 'cm-tag-mark' })
const tagMarkTheme = EditorView.baseTheme({
  '.cm-tag-mark': {
    border: '1px solid hsla(var(--tag-blue-border))',
    padding: '0px 4px',
    borderRadius: '4px',
    backgroundColor: 'hsla(var(--tag-blue-bg))',
    color: 'hsla(var(--tag-blue-text)) !important'
  },
  '.cm-tag-mark > span': {
    color: 'hsla(var(--tag-blue-text)) !important'
  }
})

function underlineRange(view: EditorView, tags: TCodeTag[] = []) {
  const doc = view.state.doc
  if (!doc.length) return Decoration.none

  let widgets: Range<Decoration>[] = []
  for (const tag of tags) {
    const { name_range, utf16_column_range } = tag
    try {
      const line = doc.lineAt(name_range.start)
      const startPos = line.from + utf16_column_range.start
      const endPos = line.from + utf16_column_range.end
      widgets.push(tagMark.range(startPos, endPos))
    } catch (e) {
      console.log(name_range.end)
      console.error(e)
    }
  }
  return Decoration.set(widgets)
}

const markTagNameExtension = (tags: TCodeTag[]) => {
  const extension = ViewPlugin.fromClass(
    class {
      underlines: DecorationSet
      constructor(view: EditorView) {
        this.underlines = underlineRange(view, tags)
      }
      update(update: ViewUpdate) {
        if (update.docChanged || update.viewportChanged) {
          this.underlines = underlineRange(update.view, tags)
        }
      }
    },
    {
      decorations: instance => {
        return instance.underlines
      }
    }
  )

  return [extension, tagMarkTheme]
}

export { markTagNameExtension, tagMarkTheme }
