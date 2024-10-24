import React, { Suspense, useContext } from 'react'

import useRouterStuff from '@/lib/hooks/use-router-stuff'
import { filename2prism } from '@/lib/language-utils'
import { cn } from '@/lib/utils'
import { ListSkeleton } from '@/components/skeleton'

import { BlobModeViewContext } from './blob-mode-view'
import { SourceCodeBrowserContext } from './source-code-browser'

const CodeEditorView = React.lazy(() => import('./code-editor-view'))
const MarkdownView = React.lazy(() => import('./markdown-view'))

interface TextFileViewProps extends React.HTMLProps<HTMLDivElement> {}

export const TextFileView: React.FC<TextFileViewProps> = ({ className }) => {
  const { searchParams } = useRouterStuff()
  const { activePath } = useContext(SourceCodeBrowserContext)
  const { textValue } = useContext(BlobModeViewContext)

  const detectedLanguage = activePath
    ? filename2prism(activePath)[0]
    : undefined
  const language = detectedLanguage ?? 'plain'
  const isMarkdown = !!textValue && language === 'markdown'
  const isPlain = searchParams.get('plain')?.toString() === '1'
  const showMarkdown = isMarkdown && !isPlain

  return (
    <div className={cn(className)}>
      <div className="rounded-b-lg border border-t-0">
        <Suspense fallback={<ListSkeleton className="p-2" />}>
          {showMarkdown ? (
            <MarkdownView value={textValue} />
          ) : (
            <CodeEditorView value={textValue ?? ''} language={language} />
          )}
        </Suspense>
      </div>
    </div>
  )
}
