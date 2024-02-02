import { Range } from '@codemirror/state'
import {
  Decoration,
  DecorationSet,
  EditorView,
  ViewPlugin,
  ViewUpdate
} from '@codemirror/view'

import type { TCodeTag } from '@/lib/types'

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

function getNameTags(view: EditorView, tags: TCodeTag[] = []) {
  const doc = view.state.doc
  const len = doc.length
  if (!len) return Decoration.none

  let widgets: Range<Decoration>[] = []
  for (const tag of tags) {
    const range = getUTF16NameRange(view.state, tag)
    try {
      if (range && range.start <= len && range.end <= len) {
        widgets.push(tagMark.range(range.start, range.end))
      }
    } catch (e) {}
  }

  if (!widgets.length) return Decoration.none
  return Decoration.set(widgets)
}

const markTagNameExtension = (tags: TCodeTag[]) => {
  const extension = ViewPlugin.fromClass(
    class {
      marks: DecorationSet
      constructor(view: EditorView) {
        this.marks = getNameTags(view, tags)
      }
      update(update: ViewUpdate) {
        if (update.docChanged || update.viewportChanged) {
          this.marks = getNameTags(update.view, tags)
        }
      }
    },
    {
      decorations: instance => {
        return instance.marks
      }
    }
  )

  return [extension, tagMarkTheme]
}

export { markTagNameExtension, tagMarkTheme }
