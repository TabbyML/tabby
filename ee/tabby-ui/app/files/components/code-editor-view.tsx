import React from 'react'
import { foldGutter } from '@codemirror/language'
import { Extension } from '@codemirror/state'
import { drawSelection, EditorView } from '@codemirror/view'
import { useTheme } from 'next-themes'

import { EXP_enable_code_browser_quick_action_bar } from '@/lib/experiment-flags'
import { TCodeTag } from '@/lib/types'
import CodeEditor, {
  CodeMirrorEditorRef
} from '@/components/codemirror/codemirror'
import { selectLinesGutter } from '@/components/codemirror/line-menu-extension/line-menu-extension'
import { markTagNameExtension } from '@/components/codemirror/name-tag-extension'
import { highlightTagExtension } from '@/components/codemirror/tag-range-highlight-extension'
import { codeTagHoverTooltip } from '@/components/codemirror/tooltip-extesion'

import { ActionBarWidgetExtension } from './action-bar-widget/action-bar-widget-extension'
import { SourceCodeBrowserContext } from './source-code-browser'
import { resolveRepositoryInfoFromPath } from './utils'

import '@/components/codemirror/line-menu-extension/line-menu.css'

interface CodeEditorViewProps {
  value: string
  language: string
}

const CodeEditorView: React.FC<CodeEditorViewProps> = ({ value, language }) => {
  const { theme } = useTheme()
  const tags: TCodeTag[] = []
  const editorRef = React.useRef<CodeMirrorEditorRef>(null)
  const { isChatEnabled, activePath } = React.useContext(
    SourceCodeBrowserContext
  )
  const filePath = React.useMemo(() => {
    const { repositoryName, basename } =
      resolveRepositoryInfoFromPath(activePath)
    return `${repositoryName}/${basename}`
  }, [activePath])

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
      selectLinesGutter,
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
          dom.style.padding = '0 8px'

          return dom
        }
      }),
      drawSelection()
    ]
    if (
      EXP_enable_code_browser_quick_action_bar.value &&
      isChatEnabled &&
      activePath
    ) {
      result.push(ActionBarWidgetExtension({ language, path: filePath }))
    }
    if (value && tags) {
      result.push(
        markTagNameExtension(tags),
        codeTagHoverTooltip(tags),
        highlightTagExtension(tags)
      )
    }

    return result
  }, [value, tags, language, editorRef.current])

  return (
    <CodeEditor
      value={value}
      theme={theme}
      language={language}
      readonly
      extensions={extensions}
      ref={editorRef}
    />
  )
}

export default CodeEditorView
