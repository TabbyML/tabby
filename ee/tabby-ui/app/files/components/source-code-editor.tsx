import React, { useContext } from 'react'
import { Extension } from '@codemirror/state'
import { EditorView } from '@codemirror/view'
import filename2prism from 'filename2prism'
import { useTheme } from 'next-themes'

import { TFileMeta } from '@/lib/types'
import { cn } from '@/lib/utils'
import { CodeMirrorEditor } from '@/components/codemirror/codemirror'
import { markTagNameExtension } from '@/components/codemirror/name-tag-extension'
import { highlightTagExtension } from '@/components/codemirror/tag-range-highlight-extension'
import { codeTagHoverTooltip } from '@/components/codemirror/tooltip-extesion'

import { SourceCodeBrowserContext } from './source-code-browser'

interface SourceCodeEditorProps {
  className?: string
  blob?: Blob
  meta?: TFileMeta
}

const SourceCodeEditor: React.FC<SourceCodeEditorProps> = ({
  className,
  blob,
  meta
}) => {
  const { activePath } = useContext(SourceCodeBrowserContext)
  const { theme } = useTheme()
  const [value, setValue] = React.useState<string>()

  const detectedLanguage = activePath
    ? filename2prism(activePath)[0]
    : undefined
  const language = (meta?.language ?? detectedLanguage) || ''
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

  React.useEffect(() => {
    const blob2Text = async (b: Blob) => {
      try {
        const v = await b.text()
        setValue(v)
      } catch (e) {
        setValue('')
      }
    }

    if (blob) {
      blob2Text(blob)
    }
  }, [blob])

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
