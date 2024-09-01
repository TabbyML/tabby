import React from 'react'
import { foldGutter } from '@codemirror/language'
import { Extension, Line } from '@codemirror/state'
import { drawSelection, EditorView } from '@codemirror/view'
import { isNil } from 'lodash-es'
import { useTheme } from 'next-themes'

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
import {
  generateEntryPath,
  isValidLineHash,
  parseLineNumberFromHash,
  viewModelToKind
} from './utils'

import './line-menu-extension/line-menu.css'

import { filename2prism } from '@/lib/language-utils'

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
  const endLineNumber = parseLineNumberFromHash(hash)?.end
  const [editorView, setEditorView] = React.useState<EditorView | null>(null)

  const {
    isChatEnabled,
    activePath,
    activeEntryInfo,
    activeRepo,
    activeRepoRef
  } = React.useContext(SourceCodeBrowserContext)
  const { basename } = activeEntryInfo
  const gitUrl = activeRepo?.gitUrl ?? ''

  const extensions = React.useMemo(() => {
    let result: Extension[] = [
      selectLinesGutter({
        onSelectLine: range => {
          if (!range) {
            updateHash('')
            return
          }

          updateHash(
            formatLineHashForCodeBrowser({
              start: range.line,
              end: range.endLine
            })
          )
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
    if (isChatEnabled && activePath && basename) {
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
        const _link = generateEntryPath(
          activeRepo,
          activeRepoRef?.ref?.commit ?? activeRepoRef?.name,
          activeEntryInfo.basename ?? '',
          viewModelToKind(activeEntryInfo.viewMode)
        )
        const link = new URL(`${window.location.origin}/files/${_link}`)

        // set hash
        if (isValidLineHash(window.location.hash)) {
          link.hash = window.location.hash
        }

        // set search
        const detectedLanguage = activeEntryInfo.basename
          ? filename2prism(activeEntryInfo.basename)[0]
          : undefined
        const isMarkdown = detectedLanguage === 'markdown'
        if (isMarkdown) {
          link.searchParams.set('plain', '1')
        }

        copyToClipboard(link.toString())
        return
      }
      if (data.action === 'copy_line') {
        if (!editorView) return
        const line = editorView.state.doc.line(lineNumber)
        let endLine: Line | undefined = undefined
        let content: string | undefined

        if (endLineNumber) {
          endLine = editorView.state.doc.line(endLineNumber)
        }
        // check if line and endLine are valid
        if (line && endLine && line.number <= endLine.number) {
          const startPos = line.from
          const endPos = endLine.to
          content = editorView.state.doc.slice(startPos, endPos).toString()
        } else if (line) {
          content = line.text
        }
        if (content) {
          copyToClipboard(content)
        }
      }
    }
    emitter.on('line_menu_action', onClickLineMenu)

    return () => {
      emitter.off('line_menu_action', onClickLineMenu)
    }
  }, [value, lineNumber, endLineNumber, editorView])

  React.useEffect(() => {
    if (!isNil(lineNumber) && editorView && value) {
      try {
        const lineInfo = editorView?.state?.doc?.line(lineNumber)
        const endLineInfo = !isNil(endLineNumber)
          ? editorView?.state?.doc?.line(endLineNumber)
          : null

        if (lineInfo) {
          const lineNumber = lineInfo.number
          const endLineNumber = endLineInfo?.number
          setSelectedLines(editorView, {
            line: lineNumber,
            endLine: endLineNumber
          })
          if (isPositionInView(editorView, lineInfo.from, 90)) {
            return
          }
          editorView.dispatch({
            effects: EditorView.scrollIntoView(lineInfo.from, {
              y: 'start',
              yMargin: 200
            })
          })
        }
      } catch (e) {}
    }

    return () => {
      if (editorView) {
        setSelectedLines(editorView, null)
      }
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
