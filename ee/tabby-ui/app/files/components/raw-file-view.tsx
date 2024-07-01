import React, { useContext } from 'react'

import { cn } from '@/lib/utils'

import { SourceCodeBrowserContext } from './source-code-browser'
import { resolveFileNameFromPath } from './utils'

interface RawFileViewProps extends React.HTMLAttributes<HTMLDivElement> {
  blob: Blob | undefined
  isImage?: boolean
}

export const RawFileView: React.FC<RawFileViewProps> = ({
  className,
  blob,
  isImage
}) => {
  const { activePath } = useContext(SourceCodeBrowserContext)

  return (
    <div className={cn(className)}>
      <div className="rounded-b-lg border border-t-0 p-2 text-center">
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
