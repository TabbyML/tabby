'use client'

import * as React from 'react'
import copy from 'copy-to-clipboard'

import { useToast } from '@/components/ui/use-toast'

export interface useCopyToClipboardProps {
  timeout?: number
  onError?: (e?: any) => void
}

export function useCopyToClipboard({
  timeout = 2000,
  onError
}: useCopyToClipboardProps) {
  const { toast } = useToast()
  const [isCopied, setIsCopied] = React.useState<Boolean>(false)

  const onCopySuccess = () => {
    setIsCopied(true)
    setTimeout(() => {
      setIsCopied(false)
    }, timeout)
  }

  const onCopyError = (error?: any) => {
    if (typeof onError === 'function') {
      onError?.(error)
      return
    }

    toast({
      title: 'Failed to copy.',
      variant: 'destructive'
    })
  }

  const copyToClipboard = (value: string) => {
    if (typeof window === 'undefined') return
    if (!value) return

    if (!!navigator.clipboard?.writeText) {
      navigator.clipboard
        .writeText(value)
        .then(onCopySuccess)
        .catch(onCopyError)
    } else {
      const copyResult = copy(value)
      if (copyResult) {
        onCopySuccess()
      } else {
        onCopyError()
      }
    }
  }

  return { isCopied, copyToClipboard }
}
