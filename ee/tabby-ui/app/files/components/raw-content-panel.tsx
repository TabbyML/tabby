import React, { useContext } from 'react'

import { cn } from '@/lib/utils'

import { SourceCodeBrowserContext } from './source-code-browser'
import { resolveFileNameFromPath } from './utils'

interface RawContentPanelProps extends React.HTMLAttributes<HTMLDivElement> {
  blob: Blob
  isImage?: boolean
}

export const RawContentPanel: React.FC<RawContentPanelProps> = ({
  className,
  blob,
  isImage
}) => {
  const { activePath } = useContext(SourceCodeBrowserContext)

  return (
    <div className={cn('text-center', className)}>
      {isImage ? (
        <img className="mx-auto" src={URL.createObjectURL(blob)} />
      ) : (
        <a
          className="text-primary hover:underline"
          download={resolveFileNameFromPath(activePath ?? '')}
          href={URL.createObjectURL(blob)}
          target="_blank"
        >
          View raw
        </a>
      )}
    </div>
  )
}
