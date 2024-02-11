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
import filename2prism from 'filename2prism'

interface SourceCodeEditorProps {
  className?: string
}

const SourceCodeEditor: React.FC<SourceCodeEditorProps> = ({ className }) => {
  const { activePath, codeMap, fileMetaMap } = useContext(
    SourceCodeBrowserContext
  )
  const { theme } = useTheme()
  const detectedLanguage = activePath ? filename2prism(activePath)[0] : undefined;
  const activeCodeContent = activePath ? codeMap?.[activePath] ?? '' : ''
  const language = activePath ? fileMetaMap?.[activePath]?.language ?? detectedLanguage: '';
  const tags = activePath ? fileMetaMap?.[activePath]?.tags : undefined

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
    if (activeCodeContent && tags) {
      result.push(
        markTagNameExtension(tags),
        codeTagHoverTooltip(tags),
        highlightTagExtension(tags)
      )
    }
    return result
  }, [activeCodeContent, tags])

  return (
    <div className={cn('source-code-browser', className)}>
      <CodeMirrorEditor
        key={activePath}
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
