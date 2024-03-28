import React from 'react'
import { Extension } from '@codemirror/state'
import { drawSelection, EditorView } from '@codemirror/view'
import { useTheme } from 'next-themes'

import { CodeBrowserQuickAction, emitter } from '@/lib/event-emitter'
import { EXP_enable_code_browser_quick_action_bar } from '@/lib/experiment-flags'
import { useIsChatEnabled } from '@/lib/hooks/use-server-info'
import { TFileMeta } from '@/lib/types'
import CodeEditor, {
  CodeMirrorEditorRef
} from '@/components/codemirror/codemirror'
import { markTagNameExtension } from '@/components/codemirror/name-tag-extension'
import { highlightTagExtension } from '@/components/codemirror/tag-range-highlight-extension'
import { codeTagHoverTooltip } from '@/components/codemirror/tooltip-extesion'

import { ActionBarWidgetExtension } from './action-bar-widget/action-bar-widget-extension'

interface CodeEditorViewProps {
  value: string
  meta?: TFileMeta
  language: string
}

const CodeEditorView: React.FC<CodeEditorViewProps> = ({
  value,
  meta,
  language
}) => {
  const { theme } = useTheme()
  const tags = meta?.tags
  const editorRef = React.useRef<CodeMirrorEditorRef>(null)
  const isChatEnabled = useIsChatEnabled()

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
    if (EXP_enable_code_browser_quick_action_bar.value && isChatEnabled) {
      result.push(ActionBarWidgetExtension())
    }
    if (value && tags) {
      result.push(
        markTagNameExtension(tags),
        codeTagHoverTooltip(tags),
        highlightTagExtension(tags)
      )
    }
    return result
  }, [value, tags, editorRef.current])

  React.useEffect(() => {
    const quickActionBarCallback = (action: CodeBrowserQuickAction) => {
      let builtInPrompt = ''
      switch (action) {
        case 'explain_detail':
          builtInPrompt = 'Explain the following code:'
          break
        case 'generate_unit-test':
          builtInPrompt = 'Generate a unit test for the following code:'
          break
        case 'generate_doc':
          builtInPrompt = 'Generate documentation for the following code:'
          break
        default:
          break
      }
      const view = editorRef.current?.editorView
      const text =
        view?.state.doc.sliceString(
          view?.state.selection.main.from,
          view?.state.selection.main.to
        ) || ''

      const initialMessage = `${builtInPrompt}\n${'```'}${
        language ?? ''
      }\n${text}\n${'```'}\n`
      if (initialMessage) {
        window.open(
          `/playground?initialMessage=${encodeURIComponent(initialMessage)}`
        )
      }
    }

    emitter.on('code_browser_quick_action', quickActionBarCallback)

    return () => {
      emitter.off('code_browser_quick_action', quickActionBarCallback)
    }
  }, [])

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
