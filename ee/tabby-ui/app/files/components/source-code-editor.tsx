import React from 'react'
import { Extension } from '@codemirror/state'
import { EditorView } from '@codemirror/view'
import { useTheme } from 'next-themes'

import { TFileMeta } from '@/lib/types'
import { cn } from '@/lib/utils'
import { CodeMirrorEditor } from '@/components/codemirror/codemirror'
import { markTagNameExtension } from '@/components/codemirror/name-tag-extension'
import { highlightTagExtension } from '@/components/codemirror/tag-range-highlight-extension'
import { codeTagHoverTooltip } from '@/components/codemirror/tooltip-extesion'

interface SourceCodeEditorProps {
  className?: string
  value: string
  meta?: TFileMeta
  language: string
}

const SourceCodeEditor: React.FC<SourceCodeEditorProps> = ({
  className,
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
    <div className={cn('source-code-browser', className)}>
      <CodeMirrorEditor
        value={value}
        theme={theme}
        language={language}
        readonly
        extensions={extensions}
      />
    </div>
  )
}

export default SourceCodeEditor
