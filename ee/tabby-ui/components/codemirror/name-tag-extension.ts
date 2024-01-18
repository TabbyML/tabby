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
    // border: '1px solid hsl(var(--border))',
    border: '1px solid rgb(147 197 253)',
    padding: '0px 4px',
    borderRadius: '4px',
    backgroundColor: 'rgb(219 234 254)',
    // backgroundColor: 'hsl(var(--primary))',
    color: 'rgb(29 78 216) !important'
    // color: 'hsl(var(--primary-foreground)) !important'
  },
  '.cm-tag-mark > span': {
    color: 'rgb(29 78 216) !important'
    // color: 'hsl(var(--primary-foreground)) !important'
  }
})

function underlineRange(view: EditorView, tags: TCodeTag[] = []) {
  const doc = view.state.doc
  if (!doc.length) return Decoration.none

  let widgets: Range<Decoration>[] = []
  for (const tag of tags) {
    const { name_range } = tag
    widgets.push(tagMark.range(name_range.start, name_range.end))
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
