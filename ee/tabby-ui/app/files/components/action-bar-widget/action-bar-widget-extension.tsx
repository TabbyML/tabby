import { StateField } from '@codemirror/state'
import type { EditorState, Extension, Transaction } from '@codemirror/state'
import { showTooltip } from '@codemirror/view'
import type { Tooltip } from '@codemirror/view'
import ReactDOM from 'react-dom/client'

import { ActionBarWidget } from './action-bar-widget'

let delayTimer: number

function ActionBarWidgetExtension(): Extension {
  return StateField.define<Tooltip | null>({
    create() {
      return null
    },
    update(value, transaction) {
      if (transaction.newSelection.main.empty) {
        clearTimeout(delayTimer)
        return null
      }
      if (transaction.selection) {
        if (shouldShowActionBarWidget(transaction)) {
          const tooltip = createActionBarWidget(transaction.state)
          // avoid flickering
          // return tooltip?.pos !== value?.pos ? tooltip : value
          return tooltip
        }

        clearTimeout(delayTimer)
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
      dom.style.border = 'none'
      const root = ReactDOM.createRoot(dom)
      dom.onclick = e => e.stopImmediatePropagation()
      // delay popup
      if (delayTimer) clearTimeout(delayTimer)
      delayTimer = window.setTimeout(() => {
        root.render(<ActionBarWidget />)
      }, 1000)

      return { dom }
    }
  }
}

function shouldShowActionBarWidget(transaction: Transaction): boolean {
  const isTextSelected =
    !!transaction.selection && !transaction.selection.main.empty
  return isTextSelected && transaction.isUserEvent('select')
}

export { ActionBarWidgetExtension }
