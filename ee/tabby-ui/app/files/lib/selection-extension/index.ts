import { Extension } from '@codemirror/state'
import { EditorView, ViewPlugin, ViewUpdate } from '@codemirror/view'

type SelectionContext =
  | {
      content: string
      startLine: number
      endLine: number
    }
  | { content: string }

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
            this.handleSelectionChange(update.view)
          }
        }

        handleSelectionChange(view: EditorView | null) {
          if (!view) {
            onSelectionChange(null)
          } else {
            const data = getActiveSelection(view)
            onSelectionChange(data)
          }
        }
      }
    )
  ]
}

export function getActiveSelection(view: EditorView): SelectionContext | null {
  const selection = view.state.selection.main
  if (selection.empty) {
    return {
      content: view.state.doc.toString()
    }
  }

  const content = view.state.sliceDoc(selection.from, selection.to)
  const startLine = view.state.doc.lineAt(selection.from).number
  const endLine = view.state.doc.lineAt(selection.to).number
  return {
    content,
    startLine,
    endLine
  }
}
