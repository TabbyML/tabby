import React from 'react'
import { type ParsedEvent, type ReconnectInterval } from 'eventsource-parser'
import { EventSourceParserStream } from 'eventsource-parser/stream'

import { AnswerRequest, AnswerResponse } from '@/lib/types'

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

  const onUpdate = (event: ParsedEvent | ReconnectInterval) => {
    if (event.type === 'event') {
      if ('data' in event) {
        const parsedChunk = JSON.parse(event.data)
        if (parsedChunk) {
          setAnswer(answer => mergeParsedAnswerData(answer, parsedChunk))
        }
      }
    }
  }

  const mergeParsedAnswerData = (
    existingData: Partial<AnswerResponse> | undefined,
    data: Partial<AnswerResponse>
  ): Partial<AnswerResponse> => {
    if (!existingData) return data

    return {
      ...existingData,
      // merge answer_delta
      answer_delta: `${existingData?.answer_delta ?? ''}${
        data?.answer_delta ?? ''
      }`,
      relevant_documents:
        data?.relevant_documents || existingData.relevant_documents,
      relevant_questions:
        data?.relevant_questions || existingData.relevant_questions
    }
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

      const eventStream = response.body
        .pipeThrough(new TextDecoderStream())
        .pipeThrough(new EventSourceParserStream())
        .getReader()

      while (true) {
        const { done, value } = await eventStream.read()
        if (done) break

        onUpdate(value)
      }
    } catch (err) {
      // Ignore abort errors as they are expected.
      if ((err as any)?.name === 'AbortError') {
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
