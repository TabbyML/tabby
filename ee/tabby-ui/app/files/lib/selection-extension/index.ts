import { Extension } from '@codemirror/state'
import { EditorView, ViewPlugin, ViewUpdate } from '@codemirror/view'

interface SelectionContext {
  content: string
  startLine: number
  endLine: number
}

export function SelectionChangeExtension(
  onSelectionChange: (fileContext: SelectionContext | null) => void
): Extension {
  return [
    ViewPlugin.fromClass(
      class {
        constructor(view: EditorView) {
          this.handleSelectionChange(view)
        }

        update(update: ViewUpdate) {
          if (update.selectionSet) {
            this.handleSelectionChange(update.view)
          } else if (update.focusChanged && !update.view.hasFocus) {
            // ignore changes if the view has lost focus
            return
          } else {
            onSelectionChange(null)
          }
          if (update.selectionSet) {
            this.handleSelectionChange(update.view)
          } else if (!update.focusChanged || update.view.hasFocus) {
            onSelectionChange(null)
          }
        }

        handleSelectionChange(view: EditorView) {
          const data = getActiveSelection(view)
          onSelectionChange(data)
        }
      }
    )
  ]
}

export function getActiveSelection(view: EditorView): SelectionContext | null {
  const selection = view.state.selection.main
  if (selection.empty) return null

  const content = view.state.sliceDoc(selection.from, selection.to)
  const startLine = view.state.doc.lineAt(selection.from).number
  const endLine = view.state.doc.lineAt(selection.to).number
  return {
    content,
    startLine,
    endLine
  }
}
