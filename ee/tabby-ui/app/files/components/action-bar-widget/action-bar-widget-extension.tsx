import type { EditorState, Extension, Transaction } from '@codemirror/state'
import { StateField } from '@codemirror/state'
import type { Tooltip } from '@codemirror/view'
import { showTooltip } from '@codemirror/view'
import ReactDOM from 'react-dom/client'

import { ActionBarWidget } from './action-bar-widget'

let delayTimer: number

interface Options {
  language?: string
  path: string
  gitUrl: string
}

function ActionBarWidgetExtension(options: Options): Extension {
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
          const tooltip = createActionBarWidget(transaction.state, options)
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

function createActionBarWidget(state: EditorState, options: Options): Tooltip {
  const { selection } = state
  const lineFrom = state.doc.lineAt(selection.main.from)
  const lineTo = state.doc.lineAt(selection.main.to)
  const isMultiline = lineFrom.number !== lineTo.number
  const pos = isMultiline ? lineTo.from : selection.main.from
  const text =
    state.doc.sliceString(state.selection.main.from, state.selection.main.to) ||
    ''

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
        root.render(
          <ActionBarWidget
            text={text}
            language={options?.language}
            lineFrom={lineFrom.number}
            lineTo={lineTo.number}
            path={options?.path}
            gitUrl={options?.gitUrl}
          />
        )
      }, 1000)

      return { dom }
    }
  }
}

function shouldShowActionBarWidget(transaction: Transaction): boolean {
  const isTextSelected =
    !!transaction.selection && !transaction.selection.main.empty
  return (
    isTextSelected &&
    transaction.isUserEvent('select') &&
    !transaction.isUserEvent('select.search')
  )
}

export { ActionBarWidgetExtension }
