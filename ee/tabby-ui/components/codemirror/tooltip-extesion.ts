import { EditorView, hoverTooltip } from '@codemirror/view'

import { TCodeTag } from '@/app/files/components/source-code-browser'

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
        const { name_range, syntax_type_name } = tag
        if (pos >= name_range.start && pos <= name_range.end) {
          return {
            pos: name_range.start,
            end: name_range.end,
            above: true,
            create(view) {
              let dom = document.createElement('div')
              dom.className = 'cm-tooltip-cursor'
              dom.textContent = `${syntax_type_name}`
              return { dom, offset: {x: 0, y: 4 } }
            }
          }
        }
      }

      return null
    }),
    cursorTooltipBaseTheme
  ]
}
