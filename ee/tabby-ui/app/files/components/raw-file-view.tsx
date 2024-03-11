import React, { useContext } from 'react'

import { cn } from '@/lib/utils'

import { BlobHeader } from './blob-header'
import { SourceCodeBrowserContext } from './source-code-browser'
import { resolveFileNameFromPath } from './utils'

interface RawContentViewProps extends React.HTMLAttributes<HTMLDivElement> {
  blob: Blob | undefined
  contentLength: number | undefined
  isImage?: boolean
}

export const RawFileView: React.FC<RawContentViewProps> = ({
  className,
  blob,
  isImage,
  contentLength
}) => {
  const { activePath } = useContext(SourceCodeBrowserContext)

  return (
    <div className={cn(className)}>
      <BlobHeader blob={blob} contentLength={contentLength} />
      <div className="text-center border rounded-b-lg border-t-0 p-2">
        {isImage ? (
          <img
            className="mx-auto"
            src={blob ? URL.createObjectURL(blob) : undefined}
          />
        ) : (
          <a
            className="text-primary hover:underline"
            download={resolveFileNameFromPath(activePath ?? '')}
            href={blob ? URL.createObjectURL(blob) : ''}
            target="_blank"
          >
            View raw
          </a>
        )}
      </div>
    </div>
  )
}
