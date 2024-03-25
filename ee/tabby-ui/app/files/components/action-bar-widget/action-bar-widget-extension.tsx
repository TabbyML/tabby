import type { EditorState, Extension, Transaction } from '@codemirror/state'
import { StateField } from '@codemirror/state'
import type { Tooltip } from '@codemirror/view'
import { showTooltip } from '@codemirror/view'
import ReactDOM from 'react-dom/client'

import { ActionBarWidget } from './action-bar-widget'

function ActionBarWidgetExtension(): Extension {
  return StateField.define<Tooltip | null>({
    create() {
      return null
    },
    update(value, transaction) {
      if (transaction.newSelection.main.empty) {
        return null
      }
      if (transaction.selection) {
        if (shouldShowActionBarWidget(transaction)) {
          const tooltip = createActionBarWidget(transaction.state)
          return tooltip?.pos !== value?.pos ? tooltip : value
        }
        return null
      }
      return value
    },
    provide: field => showTooltip.compute([field], state => state.field(field))
  })
}

function createActionBarWidget(state: EditorState): Tooltip {
  const { selection } = state
  const lineFrom = state.doc.lineAt(selection.main.from)
  const lineTo = state.doc.lineAt(selection.main.to)
  const isMultiline = lineFrom.number !== lineTo.number
  const pos = isMultiline ? lineTo.from : selection.main.from

  return {
    pos,
    above: false,
    strictSide: true,
    arrow: false,
    create() {
      const dom = document.createElement('div')
      dom.style.background = 'transparent'
      // dom.style.border = 'none'
      const root = ReactDOM.createRoot(dom)
      dom.onclick = e => e.stopImmediatePropagation()
      root.render(<ActionBarWidget />)
      return { dom }
    }
  }
}

function shouldShowActionBarWidget(transaction: Transaction): boolean {
  return (
    !!transaction.selection &&
    !transaction.selection.main.empty &&
    transaction.isUserEvent('select') &&
    !transaction.isUserEvent('select.search')
  )
}

export { ActionBarWidgetExtension }
