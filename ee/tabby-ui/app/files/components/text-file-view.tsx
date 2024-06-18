import React, { Suspense, useContext } from 'react'

import useRouterStuff from '@/lib/hooks/use-router-stuff'
import { filename2prism } from '@/lib/language-utils'
import { cn } from '@/lib/utils'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { ListSkeleton } from '@/components/skeleton'

import { BlobHeader } from './blob-header'
import { SourceCodeBrowserContext } from './source-code-browser'

const CodeEditorView = React.lazy(() => import('./code-editor-view'))
const MarkdownView = React.lazy(() => import('./markdown-view'))

interface TextFileViewProps extends React.HTMLProps<HTMLDivElement> {
  blob: Blob | undefined
  contentLength?: number
}

export const TextFileView: React.FC<TextFileViewProps> = ({
  className,
  blob,
  contentLength
}) => {
  const { searchParams, updateUrlComponents } = useRouterStuff()
  const [value, setValue] = React.useState<string>('')
  const { activePath } = useContext(SourceCodeBrowserContext)

  const detectedLanguage = activePath
    ? filename2prism(activePath)[0]
    : undefined
  const language = detectedLanguage ?? 'plain'
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
      updateUrlComponents({
        searchParams: {
          set: {
            plain: '1'
          }
        }
      })
    } else {
      updateUrlComponents({
        searchParams: {
          del: 'plain'
        }
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
      <div className="rounded-b-lg border border-t-0 py-2">
        <Suspense fallback={<ListSkeleton className="p-2" />}>
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
