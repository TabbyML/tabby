import React from 'react'
import { Extension } from '@codemirror/state'
import { drawSelection, EditorView } from '@codemirror/view'
import { useTheme } from 'next-themes'

import { EXP_enable_code_browser_quick_action_bar } from '@/lib/experiment-flags'
import { TCodeTag } from '@/lib/types'
import CodeEditor, {
  CodeMirrorEditorRef
} from '@/components/codemirror/codemirror'
import { markTagNameExtension } from '@/components/codemirror/name-tag-extension'
import { highlightTagExtension } from '@/components/codemirror/tag-range-highlight-extension'
import { codeTagHoverTooltip } from '@/components/codemirror/tooltip-extesion'

import { ActionBarWidgetExtension } from './action-bar-widget/action-bar-widget-extension'
import { SourceCodeBrowserContext } from './source-code-browser'
import { resolveRepositoryInfoFromPath } from './utils'

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
        '.cm-gutterElement': {
          padding: '0px 16px'
        },
        '.cm-gutters': {
          paddingLeft: '32px',
          backgroundColor: 'transparent',
          borderRight: 'none'
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
