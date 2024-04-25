'use client'

import { useEffect, useRef } from 'react'
import { useSearchParams } from 'next/navigation'

import { IconSpinner } from '@/components/ui/icons'

const Callback = () => {
  const searchParams = useSearchParams()
  const errorMessage = searchParams.get('errorMessage')?.toString()
  const hasSentMessage = useRef(false)

  useEffect(() => {
    if (hasSentMessage.current) return
    if (window.opener && !window.opener.closed) {
      window.opener.postMessage(
        {
          success: !errorMessage,
          errorMessage
        },
        window.location.origin
      )
      hasSentMessage.current = true
    }

    window.close()
  }, [])

  return (
    <div className="flex h-screen w-full items-center justify-center">
      <IconSpinner className="h-6 w-6" />
    </div>
  )
}

export default Callback
