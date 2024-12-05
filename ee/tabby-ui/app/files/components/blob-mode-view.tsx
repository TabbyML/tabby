import React from 'react'

import useRouterStuff from '@/lib/hooks/use-router-stuff'
import { filename2prism } from '@/lib/language-utils'
import { cn } from '@/lib/utils'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { ListSkeleton } from '@/components/skeleton'

import { BlobHeader } from './blob-header'
import { RawFileView } from './raw-file-view'
import { SourceCodeBrowserContext } from './source-code-browser'
import { TextFileView } from './text-file-view'
import { FileDisplayType } from './types'

interface BlobViewProps extends React.HTMLAttributes<HTMLDivElement> {
  blob: Blob | undefined
  contentLength: number | undefined
  fileDisplayType?: FileDisplayType
  loading?: boolean
}

type BlobModeViewContextValue = {
  textValue?: string | undefined
}

export const BlobModeViewContext =
  React.createContext<BlobModeViewContextValue>({} as BlobModeViewContextValue)

const BlobModeViewRenderer: React.FC<BlobViewProps> = ({
  className,
  blob,
  contentLength,
  fileDisplayType,
  loading
}) => {
  const { searchParams, updateUrlComponents } = useRouterStuff()
  const { activePath } = React.useContext(SourceCodeBrowserContext)
  const { textValue } = React.useContext(BlobModeViewContext)
  const isRaw = fileDisplayType === 'raw' || fileDisplayType === 'image'

  const detectedLanguage = activePath
    ? filename2prism(activePath)[0]
    : undefined
  const language = detectedLanguage ?? 'plain'
  const isMarkdown = !!textValue && language === 'markdown'
  const isPlain = searchParams.get('plain')?.toString() === '1'

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
      <div className="sticky top-0 z-10 overflow-hidden bg-background">
        <BlobHeader blob={blob} contentLength={contentLength} canCopy={!isRaw}>
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
      </div>

      {loading && !blob ? (
        <ListSkeleton className="p-2" />
      ) : isRaw ? (
        <RawFileView blob={blob} isImage={fileDisplayType === 'image'} />
      ) : (
        <TextFileView />
      )}
    </div>
  )
}

export const BlobModeView: React.FC<BlobViewProps> = props => {
  const { blob, fileDisplayType, contentLength } = props
  const [textValue, setTextValue] = React.useState<string | undefined>()
  React.useEffect(() => {
    const blob2Text = async (b: Blob) => {
      try {
        const v = await b.text()
        setTextValue(v)
      } catch (e) {
        setTextValue('')
      }
    }

    if (!!blob && fileDisplayType === 'text') {
      blob2Text(blob)
    }
  }, [blob, fileDisplayType])

  return (
    <BlobModeViewContext.Provider value={{ textValue }}>
      <BlobModeViewRenderer
        blob={blob}
        fileDisplayType={fileDisplayType}
        contentLength={contentLength}
      />
    </BlobModeViewContext.Provider>
  )
}
