'use client'

import React from 'react'
import { isNil } from 'lodash-es'
import prettyBytes from 'pretty-bytes'
import { toast } from 'sonner'

import { useCopyToClipboard } from '@/lib/hooks/use-copy-to-clipboard'
import { useIsSticky } from '@/lib/hooks/use-is-sticky'
import { cn } from '@/lib/utils'
import { Button, buttonVariants } from '@/components/ui/button'
import { IconCheck, IconCopy, IconDownload } from '@/components/ui/icons'
import { Separator } from '@/components/ui/separator'
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger
} from '@/components/ui/tooltip'

import { FileDirectoryBreadcrumb } from './file-directory-breadcrumb'
import { SourceCodeBrowserContext } from './source-code-browser'
import { resolveFileNameFromPath } from './utils'

interface BlobHeaderProps extends React.HTMLAttributes<HTMLDivElement> {
  blob: Blob | undefined
  canCopy?: boolean
  hideBlobActions?: boolean
  contentLength?: number
  lines?: number
}

export const BlobHeader: React.FC<BlobHeaderProps> = ({
  blob,
  className,
  canCopy,
  hideBlobActions,
  contentLength,
  children,
  ...props
}) => {
  const containerRef = React.useRef<HTMLDivElement>(null)
  const { activePath } = React.useContext(SourceCodeBrowserContext)
  const { isCopied, copyToClipboard } = useCopyToClipboard({ timeout: 2000 })
  const isSticky = useIsSticky(containerRef)
  const onCopy: React.MouseEventHandler<HTMLButtonElement> = async () => {
    if (isCopied || !blob) return
    try {
      const text = await blob.text()
      copyToClipboard(text)
    } catch (e) {
      toast.error('Something went wrong. Please try again.')
    }
  }

  const contentLengthText = !isNil(contentLength)
    ? prettyBytes(contentLength)
    : ''

  return (
    <div
      className={cn(
        'sticky -top-1 z-10 border',
        !isSticky && 'rounded-t-lg',
        className
      )}
      ref={containerRef}
      {...props}
    >
      {isSticky && (
        <>
          <div className="bg-secondary pl-4">
            <FileDirectoryBreadcrumb className="py-2" />
          </div>
          {isSticky && <Separator />}
        </>
      )}
      {!hideBlobActions && (
        <div
          className={cn(
            'flex items-center justify-between bg-secondary p-2 text-secondary-foreground',
            !isSticky && 'rounded-t-lg'
          )}
        >
          <div className="flex h-8 items-center gap-4 leading-8">
            {children}
            <span className="ml-2 text-sm text-muted-foreground">
              {contentLengthText}
            </span>
          </div>
          <div className="flex items-center gap-2">
            {canCopy && (
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button variant="ghost" size="icon" onClick={onCopy}>
                    {isCopied ? (
                      <IconCheck className="text-green-600" />
                    ) : (
                      <IconCopy />
                    )}
                    <span className="sr-only">Copy</span>
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Copy raw file</TooltipContent>
              </Tooltip>
            )}
            {!!blob && (
              <Tooltip>
                <TooltipTrigger asChild>
                  <a
                    className={buttonVariants({
                      variant: 'ghost',
                      size: 'icon'
                    })}
                    download={resolveFileNameFromPath(activePath ?? '')}
                    href={URL.createObjectURL(blob)}
                  >
                    <IconDownload />
                  </a>
                </TooltipTrigger>
                <TooltipContent>Download raw file</TooltipContent>
              </Tooltip>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
