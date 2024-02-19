import React, { useContext } from 'react'

import { cn } from '@/lib/utils'
import { ListSkeleton } from '@/components/skeleton'

import {
  getFileDisplayType,
  SourceCodeBrowserContext
} from './source-code-browser'

interface RawContentPanelProps extends React.HTMLAttributes<HTMLDivElement> {
  value?: any
}

export const RawContentPanel: React.FC<RawContentPanelProps> = ({
  className,
  value
}) => {
  const { activePath } = useContext(SourceCodeBrowserContext)

  const fileDisplayType = getFileDisplayType(activePath ?? '')
  const isImage = fileDisplayType.startsWith('image')

  if (!activePath || !value)
    return (
      <div className={cn(className)}>
        <ListSkeleton />
      </div>
    )

  return (
    <div className={cn('text-center', className)}>
      {isImage ? (
        <img className="mx-auto" src={URL.createObjectURL(value)} />
      ) : (
        <a
          className="text-primary hover:underline"
          download={activePath}
          href={URL.createObjectURL(value)}
          target="_blank"
        >
          view raw
        </a>
      )}
    </div>
  )
}
