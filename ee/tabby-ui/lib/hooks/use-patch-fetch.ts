import { useEffect } from 'react'
import { OpenAIStream, StreamingTextResponse } from 'ai'

import { useSession } from '../tabby/auth'

export function usePatchFetch() {
  const { data } = useSession()

  useEffect(() => {
    if (!(window as any)._originFetch) {
      ;(window as any)._originFetch = window.fetch
    }

    const fetch = (window as any)._originFetch as typeof window.fetch

    window.fetch = async function (url, options) {
      if (url !== '/api/chat') {
        return fetch(url, options)
      }

      const headers: HeadersInit = {
        'Content-Type': 'application/json'
      }

      if (data?.accessToken) {
        headers['Authorization'] = `Bearer ${data?.accessToken}`
      }

      const res = await fetch(`/v1beta/chat/completions`, {
        ...options,
        method: 'POST',
        headers
      })

      const stream = OpenAIStream(res, undefined)
      return new StreamingTextResponse(stream)
    }
  }, [data?.accessToken])
}
