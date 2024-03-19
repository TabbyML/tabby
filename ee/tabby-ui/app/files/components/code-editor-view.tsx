import React from 'react'
import { Extension } from '@codemirror/state'
import { EditorView } from '@codemirror/view'
import { useTheme } from 'next-themes'

import { TFileMeta } from '@/lib/types'
import CodeEditor from '@/components/codemirror/codemirror'
import { markTagNameExtension } from '@/components/codemirror/name-tag-extension'
import { highlightTagExtension } from '@/components/codemirror/tag-range-highlight-extension'
import { codeTagHoverTooltip } from '@/components/codemirror/tooltip-extesion'

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
      })
    ]
    if (value && tags) {
      result.push(
        markTagNameExtension(tags),
        codeTagHoverTooltip(tags),
        highlightTagExtension(tags)
      )
    }
    return result
  }, [value, tags])

  return (
    <CodeEditor
      value={value}
      theme={theme}
      language={language}
      readonly
      extensions={extensions}
    />
  )
}

export default CodeEditorView
