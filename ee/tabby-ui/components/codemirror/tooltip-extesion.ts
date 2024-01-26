import { EditorView, hoverTooltip } from '@codemirror/view'

import type { TCodeTag } from '@/lib/types'

import { getUTF16NameRange } from './utils'

const cursorTooltipBaseTheme = EditorView.baseTheme({
  '.cm-tooltip': {
    border: 'none !important'
  },
  '.cm-tooltip-cursor': {
    backgroundColor: 'hsl(var(--popover))',
    color: 'hsl(var(--popover-foreground))',
    border: 'none !important',
    padding: '2px 7px',
    borderRadius: '4px'
  }
})

export const codeTagHoverTooltip = (tags: TCodeTag[]) => {
  return [
    hoverTooltip((view, pos, side) => {
      for (const tag of tags) {
        const name_range = getUTF16NameRange(view.state, tag)
        if (!name_range) continue
        if (pos >= name_range.start && pos <= name_range.end) {
          return {
            pos,
            above: true,
            create(view) {
              let dom = document.createElement('div')
              dom.className = 'cm-tooltip-cursor'
              dom.textContent = `${tag.syntax_type_name}`
              return { dom, offset: { x: -20, y: 4 } }
            }
          }
        }
      }

      return null
    }),
    cursorTooltipBaseTheme
  ]
}
