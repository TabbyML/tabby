import React, { useContext } from 'react'
import { Extension } from '@codemirror/state'
import { EditorView } from '@codemirror/view'
import { useTheme } from 'next-themes'

import { cn } from '@/lib/utils'
import { CodeMirrorEditor } from '@/components/codemirror/codemirror'
import { markTagNameExtension } from '@/components/codemirror/name-tag-extension'
import { highlightTagExtension } from '@/components/codemirror/tag-range-highlight-extension'
import { codeTagHoverTooltip } from '@/components/codemirror/tooltip-extesion'

import { SourceCodeBrowserContext } from './source-code-browser'

interface SourceCodeEditorProps {
  className?: string
}

const SourceCodeEditor: React.FC<SourceCodeEditorProps> = ({ className }) => {
  const { activePath, codeMap, fileMetaMap } = useContext(
    SourceCodeBrowserContext
  )
  const { theme } = useTheme()
  const activeCodeContent = activePath ? codeMap?.[activePath] ?? '' : ''
  const language = activePath ? fileMetaMap?.[activePath]?.language ?? '' : ''
  const tags = React.useMemo(() => {
    return activePath ? fileMetaMap?.[activePath]?.tags || [] : []
  }, [activePath, fileMetaMap])

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
          marginLeft: '32px',
          backgroundColor: 'transparent',
          borderRight: 'none'
        }
      })
    ]
    if (activeCodeContent && tags) {
      result.push(
        codeTagHoverTooltip(tags),
        markTagNameExtension(tags),
        highlightTagExtension(tags)
      )
    }
    return result
  }, [activeCodeContent, tags])

  return (
    <div
      className={cn('source-code-browser h-full overflow-y-auto', className)}
    >
      <CodeMirrorEditor
        value={activeCodeContent}
        theme={theme}
        language={language}
        readonly
        extensions={extensions}
      />
    </div>
  )
}

export default SourceCodeEditor
