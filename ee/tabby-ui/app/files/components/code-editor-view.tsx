import React from 'react'
import { foldGutter } from '@codemirror/language'
import { Extension } from '@codemirror/state'
import { drawSelection, EditorView } from '@codemirror/view'
import { isNaN, isNil } from 'lodash-es'
import { useTheme } from 'next-themes'

import { EXP_enable_code_browser_quick_action_bar } from '@/lib/experiment-flags'
import { useCopyToClipboard } from '@/lib/hooks/use-copy-to-clipboard'
import { useHash } from '@/lib/hooks/use-hash'
import { TCodeTag } from '@/lib/types'
import { formatLineHashForCodeBrowser } from '@/lib/utils'
import CodeEditor from '@/components/codemirror/codemirror'
import { markTagNameExtension } from '@/components/codemirror/name-tag-extension'
import { highlightTagExtension } from '@/components/codemirror/tag-range-highlight-extension'
import { codeTagHoverTooltip } from '@/components/codemirror/tooltip-extesion'

import { emitter, LineMenuActionEventPayload } from '../lib/event-emitter'
import { ActionBarWidgetExtension } from './action-bar-widget/action-bar-widget-extension'
import {
  selectLinesGutter,
  setSelectedLines
} from './line-menu-extension/line-menu-extension'
import { SourceCodeBrowserContext } from './source-code-browser'
import { parseLineNumberFromHash } from './utils'

import './line-menu-extension/line-menu.css'

interface CodeEditorViewProps {
  value: string
  language: string
}

const CodeEditorView: React.FC<CodeEditorViewProps> = ({ value, language }) => {
  const { theme } = useTheme()
  const tags: TCodeTag[] = React.useMemo(() => {
    return []
  }, [])
  const { copyToClipboard } = useCopyToClipboard({})
  const [hash, updateHash] = useHash()
  const lineNumber = parseLineNumberFromHash(hash)?.start
  const [editorView, setEditorView] = React.useState<EditorView | null>(null)

  const { isChatEnabled, activePath, activeEntryInfo, activeRepo } =
    React.useContext(SourceCodeBrowserContext)
  const { basename } = activeEntryInfo
  const gitUrl = activeRepo?.gitUrl ?? ''

  const extensions = React.useMemo(() => {
    let result: Extension[] = [
      EditorView.baseTheme({
        '.cm-scroller': {
          fontSize: '14px'
        },
        '.cm-gutters': {
          backgroundColor: 'transparent',
          borderRight: 'none'
        }
      }),
      selectLinesGutter({
        onSelectLine: v => {
          if (v === -1 || isNaN(v)) return
          // todo support multi lines
          updateHash(formatLineHashForCodeBrowser({ start: v }))
        }
      }),
      foldGutter({
        markerDOM(open) {
          const dom = document.createElement('div')
          dom.style.cursor = 'pointer'
          if (open) {
            dom.innerHTML =
              '<svg aria-hidden="true" focusable="false" role="img" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" style="display: inline-block; user-select: none; vertical-align: text-bottom; overflow: visible;"><path d="M12.78 5.22a.749.749 0 0 1 0 1.06l-4.25 4.25a.749.749 0 0 1-1.06 0L3.22 6.28a.749.749 0 1 1 1.06-1.06L8 8.939l3.72-3.719a.749.749 0 0 1 1.06 0Z"></path></svg>'
          } else {
            dom.innerHTML =
              '<svg aria-hidden="true" focusable="false" role="img" viewBox="0 0 16 16" width="16" height="16" fill="currentColor" style="display: inline-block; user-select: none; vertical-align: text-bottom; overflow: visible;"><path d="M6.22 3.22a.75.75 0 0 1 1.06 0l4.25 4.25a.75.75 0 0 1 0 1.06l-4.25 4.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042L9.94 8 6.22 4.28a.75.75 0 0 1 0-1.06Z"></path></svg>'
          }
          return dom
        }
      }),
      drawSelection()
    ]
    if (
      EXP_enable_code_browser_quick_action_bar.value &&
      isChatEnabled &&
      activePath &&
      basename
    ) {
      result.push(
        ActionBarWidgetExtension({ language, path: basename, gitUrl })
      )
    }
    if (value && tags) {
      result.push(
        markTagNameExtension(tags),
        codeTagHoverTooltip(tags),
        highlightTagExtension(tags)
      )
    }

    return result
  }, [value, tags, language])

  React.useEffect(() => {
    const onClickLineMenu = (data: LineMenuActionEventPayload) => {
      if (typeof lineNumber !== 'number') return
      if (data.action === 'copy_permalink') {
        copyToClipboard(window.location.href)
        return
      }
      if (data.action === 'copy_line') {
        const lineObject = editorView?.state?.doc?.line(lineNumber)
        if (lineObject) {
          copyToClipboard(lineObject.text)
        }
      }
    }
    emitter.on('line_menu_action', onClickLineMenu)

    return () => {
      emitter.off('line_menu_action', onClickLineMenu)
    }
  }, [value, lineNumber])

  React.useEffect(() => {
    if (!isNil(lineNumber) && editorView && value) {
      try {
        const lineInfo = editorView?.state?.doc?.line(lineNumber)

        if (lineInfo) {
          const pos = lineInfo.from
          setSelectedLines(editorView, pos)
          if (isPositionInView(editorView, pos, 90)) {
            return
          }
          editorView.dispatch({
            effects: EditorView.scrollIntoView(pos, {
              y: 'start',
              yMargin: 200
            })
          })
        }
      } catch (e) {}
    }
  }, [value, lineNumber, editorView])

  return (
    <CodeEditor
      value={value}
      theme={theme}
      language={language}
      readonly
      extensions={extensions}
      viewDidUpdate={view => setEditorView(view)}
    />
  )
}

function isPositionInView(
  view: EditorView,
  pos: number,
  offsetTop: number = 0
) {
  const node = view.domAtPos(pos).node
  const lineElement =
    node.nodeType === 3 ? node.parentElement : (node as HTMLElement)
  if (lineElement) {
    const rect = lineElement.getBoundingClientRect()
    const viewportHeight =
      window.innerHeight || document.documentElement.clientHeight
    return rect.top >= offsetTop && rect.bottom <= viewportHeight
  }

  return false
}

export default CodeEditorView
