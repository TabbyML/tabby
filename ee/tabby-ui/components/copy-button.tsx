'use client'

import * as React from 'react'

import { useCopyToClipboard } from '@/lib/hooks/use-copy-to-clipboard'
import { Button, type ButtonProps } from '@/components/ui/button'

import { IconCheck, IconCopy } from './ui/icons'

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
      {isCopied ? <IconCheck className="text-green-600" /> : <IconCopy />}
      <span className="sr-only">Copy</span>
    </Button>
  )
}
