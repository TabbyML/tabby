import { Range } from '@codemirror/state'
import {
  Decoration,
  DecorationSet,
  EditorView,
  ViewPlugin,
  ViewUpdate
} from '@codemirror/view'

import { TCodeTag } from '@/app/files/components/source-code-browser'

import { getUTF16NameRange } from './utils'

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
    const range = getUTF16NameRange(view.state, tag)
    widgets.push(tagMark.range(range.start, range.end))
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
