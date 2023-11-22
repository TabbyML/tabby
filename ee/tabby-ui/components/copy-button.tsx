'use client'

import * as React from 'react'

import { Button, type ButtonProps } from '@/components/ui/button'
import { IconCheck, IconCopy } from './ui/icons'
import { useCopyToClipboard } from '@/lib/hooks/use-copy-to-clipboard'

interface CopyButtonProps extends ButtonProps {
  value: string
}

export function CopyButton({ className, value, ...props }: CopyButtonProps) {
  const { isCopied, copyToClipboard } = useCopyToClipboard({ timeout: 2000 })

  const onCopy = () => {
    if (isCopied) return
    copyToClipboard(value)
  }

  if (!value) return null

  return (
    <Button
      variant="ghost"
      size="icon"
      className={className}
      onClick={onCopy}
      {...props}
    >
      {isCopied ? <IconCheck /> : <IconCopy />}
      <span className="sr-only">Copy</span>
    </Button>
  )
}
