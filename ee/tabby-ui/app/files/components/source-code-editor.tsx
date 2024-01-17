import React, { useContext } from 'react'
import { useTheme } from 'next-themes'

import { cn } from '@/lib/utils'
import { CodeMirrorEditor } from '@/components/codemirror/codemirror'
import { underlineTagNameExtension } from '@/components/codemirror/tag-name-underline-extension'
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
    if (activeCodeContent && tags) {
      return [
        codeTagHoverTooltip(tags),
        underlineTagNameExtension(tags),
        highlightTagExtension(tags)
      ]
    }
    return undefined
  }, [activeCodeContent, tags])

  return (
    <div className={cn('h-full overflow-y-auto', className)}>
      <CodeMirrorEditor
        value={activeCodeContent}
        theme={theme}
        language={language}
        tags={tags}
        readonly={false}
        extensions={extensions}
      />
    </div>
  )
}

export default SourceCodeEditor
