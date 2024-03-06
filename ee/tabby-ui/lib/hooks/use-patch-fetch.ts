import { useEffect } from 'react'
import { OpenAIStream, StreamingTextResponse } from 'ai'

import fetcher from '../tabby/fetcher'

export function usePatchFetch() {
  useEffect(() => {
    if (!window._originFetch) {
      window._originFetch = window.fetch
    }

    const fetch = window._originFetch

    window.fetch = async function (url, options) {
      if (url !== '/api/chat') {
        return fetch(url, options)
      }

      const headers: HeadersInit = {
        'Content-Type': 'application/json'
      }

      const res = await fetcher(`/v1beta/chat/completions`, {
        ...options,
        method: 'POST',
        headers,
        customFetch: fetch,
        responseFormatter(response) {
          const stream = OpenAIStream(response, undefined)
          return new StreamingTextResponse(stream)
        }
      })

      return res
    }

    return () => {
      if (window?._originFetch) {
        window.fetch = window._originFetch
        window._originFetch = undefined
      }
    }
  }, [])
}
