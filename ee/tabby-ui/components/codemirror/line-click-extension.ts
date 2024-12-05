import { EditorView } from '@codemirror/view'

const lineClickTheme = EditorView.theme({
  '.cm-line': {
    cursor: 'pointer'
  },
  '.cm-line:hover': {
    opacity: '60%'
  }
})

export function lineClickExtension(
  onLineClick: (lineNumber: number, event: MouseEvent) => void
) {
  return [
    lineClickTheme,
    EditorView.domEventHandlers({
      mousedown(event, view) {
        const pos = view.posAtCoords({ x: event.clientX, y: event.clientY })
        if (pos != null) {
          const line = view.state.doc.lineAt(pos)
          onLineClick(line.number, event)
        }
      }
    })
  ]
}
