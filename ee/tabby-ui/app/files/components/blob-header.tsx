import React from 'react'
import { Button, buttonVariants } from '@/components/ui/button'
import { IconCheck, IconCopy, IconDownload } from '@/components/ui/icons'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { useCopyToClipboard } from '@/lib/hooks/use-copy-to-clipboard'
import { cn } from '@/lib/utils'
import { SourceCodeBrowserContext } from './source-code-browser'
import { resolveFileNameFromPath } from './utils'
import { toast } from 'sonner'

interface BlobHeaderProps extends React.HTMLAttributes<HTMLDivElement> {
  blob: Blob | undefined
  canCopy?: boolean
}

const BlobHeader: React.FC<BlobHeaderProps> = ({
  blob,
  className,
  canCopy,
  ...props
}) => {

  const { activePath } = React.useContext(SourceCodeBrowserContext)
  const { isCopied, copyToClipboard } = useCopyToClipboard({ timeout: 2000 })

  const onCopy: React.MouseEventHandler<HTMLButtonElement> = async () => {
    if (isCopied || !blob) return
    try {
      const text = await blob.text()
      copyToClipboard(text)
    } catch (e) {
      toast.error('Something went wrong. Please try again.')
    }
  }

  return (
    <div className={cn('sticky top-0', className)} {...props}>
      {/* todo put the breadcrumb here */}
      <div className='flex items-center justify-between'>
        <div>Code</div>
        <div className='flex items-center gap-2'>
          {canCopy && (
            <Tooltip>
              <TooltipTrigger>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={onCopy}
                >
                  {isCopied ? <IconCheck className="text-green-600" /> : <IconCopy />}
                  <span className="sr-only">Copy</span>
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                Copy raw file
              </TooltipContent>
            </Tooltip>
          )}
          {!!blob && (
            <Tooltip>
              <TooltipTrigger>
                <a className={buttonVariants({ variant: 'ghost', size: 'icon' })} download={resolveFileNameFromPath(activePath ?? '')} href={URL.createObjectURL(blob)}>
                  <IconDownload />
                </a>
              </TooltipTrigger>
              <TooltipContent>
                Download raw file
              </TooltipContent>
            </Tooltip>
          )}
        </div>
      </div>
    </div>
  )
}