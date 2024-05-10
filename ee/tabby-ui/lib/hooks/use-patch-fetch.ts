import { useEffect } from 'react'
import {
  Message,
  OpenAIStream,
  OpenAIStreamCallbacks,
  StreamingTextResponse
} from 'ai'

import fetcher from '../tabby/fetcher'

interface PatchFetchOptions extends OpenAIStreamCallbacks {}

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

      const res = await fetcher(`/v1/chat/completions`, {
        ...requestInit,
        body: mergeMessagesByRole(requestInit?.body),
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

function mergeMessagesByRole(body: BodyInit | null | undefined) {
  if (typeof body !== 'string') return body
  try {
    const bodyObject = JSON.parse(body)
    let messages: Message[] = bodyObject.messages?.slice()
    if (Array.isArray(messages) && messages.length > 1) {
      let previewCursor = 0
      let curCursor = 1
      while (curCursor < messages.length) {
        let prevMessage = messages[previewCursor]
        let curMessage = messages[curCursor]
        if (curMessage.role === prevMessage.role) {
          messages = [
            ...messages.slice(0, previewCursor),
            {
              ...prevMessage,
              content: [prevMessage.content, curMessage.content].join('\n')
            },
            ...messages.slice(curCursor + 1)
          ]
        } else {
          previewCursor = curCursor++
        }
      }
      return JSON.stringify({
        ...bodyObject,
        messages
      })
    } else {
      return body
    }
  } catch (e) {
    return body
  }
}
