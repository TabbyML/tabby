import { Range } from '@codemirror/state'
import {
  Decoration,
  DecorationSet,
  EditorView,
  ViewPlugin,
  ViewUpdate
} from '@codemirror/view'

import { TCodeTag } from '@/app/browser/components/source-code-browser'

const underlineMark = Decoration.mark({ class: 'cm-underline' })

const underlineTheme = EditorView.baseTheme({
  '.cm-underline': {
    textDecoration: 'underline',
    textUnderlineOffset: '3px',
    textDecorationThickness: '2px',
    textDecorationColor: 'hsl(var(--primary))'
  }
})

function underlineRange(view: EditorView, tags: TCodeTag[] = []) {
  const doc = view.state.doc
  if (!doc.length) return Decoration.none

  let widgets: Range<Decoration>[] = []
  for (const tag of tags) {
    const { name_range } = tag
    widgets.push(underlineMark.range(name_range.start, name_range.end))
  }
  return Decoration.set(widgets)
}

const underlineTagNameExtension = (tags: TCodeTag[]) => {
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

  return [extension, underlineTheme]
}

export { underlineTagNameExtension, underlineTheme }
