import { EditorView, hoverTooltip } from '@codemirror/view'

import { TCodeTag } from '@/app/files/components/source-code-browser'

const cursorTooltipBaseTheme = EditorView.baseTheme({
  '.cm-tooltip': {
    border: 'none !important'
  },
  '.cm-tooltip-cursor': {
    backgroundColor: 'hsl(var(--primary))',
    color: 'hsl(var(--primary-foreground))',
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
              const nameText = view.state.doc
                .slice(name_range.start, name_range.end)
                .toString()
              dom.textContent = `${syntax_type_name}: ${nameText}`
              return { dom }
            }
          }
        }
      }

      return null
    }),
    cursorTooltipBaseTheme
  ]
}
