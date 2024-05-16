import React from 'react'
import { Message, OpenAIStream, StreamingTextResponse } from 'ai'

interface PatchFetchOptions {
  api: string
  fetcher?: typeof fetch
}

export function usePatchFetch({ api, fetcher }: PatchFetchOptions) {
  React.useEffect(() => {
    if (!window._originFetch) {
      window._originFetch = window.fetch
    }

    const fetch = fetcher || window._originFetch

    window.fetch = async function (url, requestInit) {
      if (url !== '/api/chat') {
        return window._originFetch!(url, requestInit)
      }

      return fetch(api, {
        ...requestInit,
        body: mergeMessagesByRole(requestInit?.body),
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...requestInit?.headers
        }
      }).then(response => {
        if (!response?.ok) {
          throw new Error(String(response.status))
        }
        const stream = OpenAIStream(response)
        return new StreamingTextResponse(stream)
      })
    }

    return () => {
      if (window?._originFetch) {
        window.fetch = window._originFetch
        window._originFetch = undefined
      }
    }
  }, [api])
}

export function mergeMessagesByRole(body: BodyInit | null | undefined) {
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
