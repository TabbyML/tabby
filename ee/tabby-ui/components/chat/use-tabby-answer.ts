import React from 'react'
import { mergeWith } from 'lodash-es'

import { AnswerRequest, AnswerResponse } from '@/lib/types'

const NEWLINE = '\n'.charCodeAt(0)

interface UseTabbyAnswerOptions {
  onError?: (err: Error) => void
  api?: string
  fetcher?: typeof fetch
  headers?: Record<string, string> | Headers
}

export function useTabbyAnswer({
  api = '/v1beta/answer',
  onError,
  headers,
  fetcher
}: UseTabbyAnswerOptions) {
  const [isLoading, setIsLoading] = React.useState(false)
  const [answer, setAnswer] = React.useState<AnswerResponse | undefined>()
  const [error, setError] = React.useState<Error | undefined>()
  // Abort controller to cancel the current API call.
  const abortControllerRef = React.useRef<AbortController | null>(null)

  const stop = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
      abortControllerRef.current = null
    }
  }

  const mergeParsedData = (
    data1: Partial<AnswerResponse>,
    data2: Partial<AnswerResponse>
  ): Partial<AnswerResponse> => {
    return mergeWith({}, data1, data2, (obj, src) => {
      if (typeof obj === 'string' && typeof src === 'string') {
        return obj + src
      }
    })
  }

  const triggerRequest = async (data: AnswerRequest) => {
    try {
      setIsLoading(true)
      setError(undefined)
      setAnswer(undefined)
      const abortController = new AbortController()
      abortControllerRef.current = abortController

      const fetch = fetcher ?? window.fetch

      const response = await fetch(api, {
        method: 'POST',
        body: JSON.stringify(data),
        headers: {
          'Content-Type': 'application/json',
          ...headers
        },
        signal: abortController?.signal
      }).catch(err => {
        throw err
      })

      if (!response?.ok) {
        throw new Error(String(response.status))
      }

      if (response.body == null) {
        throw new Error('The response body is empty')
      }

      for await (const line of readDataStream(response.body.getReader(), {
        isAborted: () => abortController.signal.aborted
      })) {
        if (line.startsWith('data: ')) {
          const jsonData = line.replace('data: ', '')
          const parsedData = JSON.parse(jsonData)
          setAnswer(answer => mergeParsedData(answer ?? {}, parsedData))
        }
      }
    } catch (err) {
      // Ignore abort errors as they are expected.
      if ((err as any).name === 'AbortError') {
        abortControllerRef.current = null
        return null
      }

      if (onError && err instanceof Error) {
        onError(err)
      }

      setError(err as Error)
    } finally {
      setIsLoading(false)
    }
  }

  return {
    isLoading,
    answer,
    error,
    setError,
    triggerRequest,
    stop
  }
}

function concatChunks(chunks: Uint8Array[], totalLength: number) {
  const concatenatedChunks = new Uint8Array(totalLength)

  let offset = 0
  for (const chunk of chunks) {
    concatenatedChunks.set(chunk, offset)
    offset += chunk.length
  }
  chunks.length = 0

  return concatenatedChunks
}

async function* readDataStream(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  {
    isAborted
  }: {
    isAborted?: () => boolean
  } = {}
): AsyncGenerator<string> {
  const decoder = new TextDecoder()
  const chunks: Uint8Array[] = []
  let totalLength = 0

  while (true) {
    const { value, done } = await reader.read()

    if (done) {
      break
    }

    if (value) {
      chunks.push(value)
      totalLength += value.length
      if (value[value.length - 1] !== NEWLINE) {
        // if the last character is not a newline, we have not read the whole JSON value
        continue
      }
    }

    if (chunks.length === 0) {
      break // we have reached the end of the stream
    }

    const concatenatedChunks = concatChunks(chunks, totalLength)
    totalLength = 0

    const streamParts = decoder
      .decode(concatenatedChunks, { stream: true })
      .split('\n')
      .filter(line => line !== '') // splitting leaves an empty string at the end

    for (const streamPart of streamParts) {
      yield streamPart
    }

    // The request has been aborted, stop reading the stream.
    if (isAborted?.()) {
      reader.cancel()
      break
    }
  }
}
