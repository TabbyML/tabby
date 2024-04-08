import { useEffect } from 'react'
import { OpenAIStream, OpenAIStreamCallbacks, StreamingTextResponse } from 'ai'

import fetcher from '../tabby/fetcher'

interface PatchFetchOptions extends OpenAIStreamCallbacks {
  processRequestBody?: (
    body: BodyInit | null | undefined
  ) => BodyInit | null | undefined
}

export function usePatchFetch(options?: PatchFetchOptions) {
  useEffect(() => {
    if (!window._originFetch) {
      window._originFetch = window.fetch
    }

    const fetch = window._originFetch

    window.fetch = async function (url, requestInit) {
      if (url !== '/api/chat') {
        return fetch(url, requestInit)
      }

      const headers: HeadersInit = {
        'Content-Type': 'application/json'
      }

      const processRequestBody = (body: BodyInit | null | undefined) => {
        if (options?.processRequestBody) {
          return options.processRequestBody(body)
        }
        return body
      }

      const res = await fetcher(`/v1beta/chat/completions`, {
        ...requestInit,
        body: processRequestBody(requestInit?.body),
        method: 'POST',
        headers,
        customFetch: fetch,
        responseFormatter(response) {
          const stream = OpenAIStream(response, options)
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
