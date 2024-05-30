'use client'

import * as React from 'react'
import copy from 'copy-to-clipboard'
import { toast } from 'sonner'

export interface useCopyToClipboardProps {
  timeout?: number
  onError?: (e?: any) => void
  onCopyContent?: (value: string) => void
}

export function useCopyToClipboard({
  timeout = 2000,
  onError,
  onCopyContent
}: useCopyToClipboardProps) {
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

    toast.error('Failed to copy.')
  }

  const copyToClipboard = (value: string) => {
    if (typeof window === 'undefined') return
    if (!value) return

    if (onCopyContent) {
      onCopyContent(value)
      onCopySuccess()
      return
    }

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
