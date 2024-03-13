import React, { Suspense, useContext } from 'react'
import filename2prism from 'filename2prism'

import useRouterStuff from '@/lib/hooks/use-router-stuff'
import { TFileMeta } from '@/lib/types'
import { cn } from '@/lib/utils'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { ListSkeleton } from '@/components/skeleton'

import { BlobHeader } from './blob-header'
import { SourceCodeBrowserContext } from './source-code-browser'

const CodeEditorView = React.lazy(() => import('./code-editor-view'))
const MarkdownView = React.lazy(() => import('./markdown-view'))

interface TextFileViewProps extends React.HTMLProps<HTMLDivElement> {
  blob: Blob | undefined
  meta: TFileMeta | undefined
  contentLength?: number
}

export const TextFileView: React.FC<TextFileViewProps> = ({
  className,
  blob,
  meta,
  contentLength
}) => {
  const { searchParams, updateSearchParams } = useRouterStuff()
  const [value, setValue] = React.useState<string>('')
  const { activePath } = useContext(SourceCodeBrowserContext)

  const detectedLanguage = activePath
    ? filename2prism(activePath)[0]
    : undefined
  const language = (meta?.language ?? detectedLanguage) || ''
  const isMarkdown = !!value && language === 'markdown'
  const isPlain = searchParams.get('plain')?.toString() === '1'
  const showMarkdown = isMarkdown && !isPlain

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

  const onToggleMarkdownView = (value: string) => {
    if (value === '1') {
      updateSearchParams({
        set: {
          plain: '1'
        }
      })
    } else {
      updateSearchParams({
        del: 'plain'
      })
    }
  }

  return (
    <div className={cn(className)}>
      <BlobHeader blob={blob} contentLength={contentLength} canCopy>
        {isMarkdown && (
          <Tabs
            value={isPlain ? '1' : '0'}
            onValueChange={onToggleMarkdownView}
          >
            <TabsList>
              <TabsTrigger value="0">Preview</TabsTrigger>
              <TabsTrigger value="1">Code</TabsTrigger>
            </TabsList>
          </Tabs>
        )}
      </BlobHeader>
      <div className="rounded-b-lg border border-t-0 p-2">
        <Suspense fallback={<ListSkeleton />}>
          {showMarkdown ? (
            <MarkdownView value={value} />
          ) : (
            <CodeEditorView value={value} language={language} />
          )}
        </Suspense>
      </div>
    </div>
  )
}
