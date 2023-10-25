import { type Message } from 'ai/react'
import { StreamingTextResponse } from 'ai'
import { TabbyStream } from '@/lib/tabby-stream'
import { useEffect } from 'react'

const serverUrl =
  process.env.NEXT_PUBLIC_TABBY_SERVER_URL || ''

export function usePatchFetch() {
  useEffect(() => {
    const fetch = window.fetch

    window.fetch = async function (url, options) {
      if (url !== '/api/chat') {
        return fetch(url, options)
      }

      const { messages } = JSON.parse(options!.body as string)
      const res = await fetch(`${serverUrl}/v1beta/chat/completions`, {
        ...options,
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
      })

      const stream = TabbyStream(res, undefined)
      return new StreamingTextResponse(stream)
    }
  }, [])
}

