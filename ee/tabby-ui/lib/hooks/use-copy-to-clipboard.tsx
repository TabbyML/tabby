'use client'

import * as React from 'react'
import copy from 'copy-to-clipboard'

export interface useCopyToClipboardProps {
  timeout?: number
}

export function useCopyToClipboard({
  timeout = 2000
}: useCopyToClipboardProps) {
  const [isCopied, setIsCopied] = React.useState<Boolean>(false)

  const onCopySuccess = () => {
    setIsCopied(true)
    setTimeout(() => {
      setIsCopied(false)
    }, timeout)
  }

  const copyToClipboard = (value: string) => {
    if (typeof window === 'undefined') return
    if (!value) return

    if (!!navigator.clipboard?.writeText) {
      navigator.clipboard
        .writeText(value)
        .then(onCopySuccess)
        .catch(() => {})
    } else {
      const copyResult = copy(value)
      if (copyResult) {
        onCopySuccess()
      }
    }

    // When component inside an iframe sandbox(VSCode)
    // We need to notify parent environment to handle the copy event
    parent.postMessage(
      {
        action: 'copy',
        data: value
      },
      '*'
    )
    onCopySuccess()
  }

  return { isCopied, copyToClipboard }
}
