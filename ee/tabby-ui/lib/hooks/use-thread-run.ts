import React from 'react'
import { pickBy } from 'lodash-es'
import { OperationContext } from 'urql'

import { graphql } from '@/lib/gql/generates'

import {
  CreateMessageInput,
  ThreadRunItem,
  ThreadRunOptionsInput
} from '../gql/generates/graphql'
import { client, useMutation } from '../tabby/gql'
import { useLatest } from './use-latest'

interface UseThreadRunOptions {
  onError?: (err: Error) => void
  threadId?: string
  headers?: Record<string, string> | Headers
  onAssistantMessageCompleted?: (
    threadId: string,
    threadRunId: ThreadRunItem | undefined
  ) => void
}

const CreateThreadAndRunSubscription = graphql(/* GraphQL */ `
  subscription CreateThreadAndRun($input: CreateThreadAndRunInput!) {
    createThreadAndRun(input: $input) {
      threadCreated
      threadUserMessageCreated
      threadAssistantMessageCreated
      threadRelevantQuestions
      threadAssistantMessageAttachmentsCode {
        code {
          gitUrl
          filepath
          language
          content
          startLine
        }
        scores {
          rrf
          bm25
          embedding
        }
      }
      threadAssistantMessageAttachmentsDoc {
        doc {
          title
          link
          content
        }
        score
      }
      threadAssistantMessageContentDelta
      threadAssistantMessageCompleted
    }
  }
`)

const CreateThreadRunSubscription = graphql(/* GraphQL */ `
  subscription CreateThreadRun($input: CreateThreadRunInput!) {
    createThreadRun(input: $input) {
      threadCreated
      threadUserMessageCreated
      threadAssistantMessageCreated
      threadRelevantQuestions
      threadAssistantMessageAttachmentsCode {
        code {
          gitUrl
          filepath
          language
          content
          startLine
        }
        scores {
          rrf
          bm25
          embedding
        }
      }
      threadAssistantMessageAttachmentsDoc {
        doc {
          title
          link
          content
        }
        score
      }
      threadAssistantMessageContentDelta
      threadAssistantMessageCompleted
    }
  }
`)

const DeleteThreadMessagePairMutation = graphql(/* GraphQL */ `
  mutation DeleteThreadMessagePair(
    $threadId: ID!
    $userMessageId: ID!
    $assistantMessageId: ID!
  ) {
    deleteThreadMessagePair(
      threadId: $threadId
      userMessageId: $userMessageId
      assistantMessageId: $assistantMessageId
    )
  }
`)

export function useThreadRun({
  threadId: propsThreadId,
  headers,
  onAssistantMessageCompleted
}: UseThreadRunOptions) {
  const [threadId, setThreadId] = React.useState<string | undefined>(
    propsThreadId
  )
  const unsubscribeFn = React.useRef<(() => void) | undefined>()
  const [isLoading, setIsLoading] = React.useState(false)
  const [threadRunItem, setThreadRunItem] = React.useState<
    ThreadRunItem | undefined
  >()
  const [error, setError] = React.useState<Error | undefined>()

  const operationContext: Partial<OperationContext> = React.useMemo(() => {
    if (headers) {
      return {
        fetchOptions: {
          headers
        }
      }
    }
    return {}
  }, [headers])

  const combineThreadRunData = (
    existingData: ThreadRunItem | undefined,
    data: ThreadRunItem | undefined
  ): ThreadRunItem | undefined => {
    if (!data) return data

    return {
      ...existingData,
      ...pickBy(data, v => v !== null),
      threadAssistantMessageContentDelta: `${
        existingData?.threadAssistantMessageContentDelta ?? ''
      }${data?.threadAssistantMessageContentDelta ?? ''}`
    }
  }

  const stop = useLatest((silent?: boolean) => {
    unsubscribeFn.current?.()
    unsubscribeFn.current = undefined
    setIsLoading(false)

    if (!silent && threadId) {
      onAssistantMessageCompleted?.(threadId, threadRunItem)
    }
  })

  React.useEffect(() => {
    if (propsThreadId && propsThreadId !== threadId) {
      setThreadId(propsThreadId)
    }
  }, [propsThreadId])

  const createThreadAndRun = (
    userMessage: CreateMessageInput,
    options?: ThreadRunOptionsInput
  ) => {
    const { unsubscribe } = client
      .subscription(
        CreateThreadAndRunSubscription,
        {
          input: {
            thread: {
              userMessage
            },
            options
          }
        },
        operationContext
      )
      .subscribe(res => {
        if (res?.error) {
          setIsLoading(false)
          setError(res.error)
          unsubscribe()
          return
        }

        if (res?.data?.createThreadAndRun?.threadAssistantMessageCompleted) {
          stop.current()
        }

        const threadIdFromData = res.data?.createThreadAndRun?.threadCreated
        if (!!threadIdFromData && threadIdFromData !== threadId) {
          setThreadId(threadIdFromData)
        }

        setThreadRunItem(prevData =>
          combineThreadRunData(prevData, res.data?.createThreadAndRun)
        )
      })

    return unsubscribe
  }

  const createThreadRun = (
    userMessage: CreateMessageInput,
    options?: ThreadRunOptionsInput
  ) => {
    if (!threadId) return
    const { unsubscribe } = client
      .subscription(
        CreateThreadRunSubscription,
        {
          input: {
            threadId,
            additionalUserMessage: userMessage,
            options
          }
        },
        operationContext
      )
      .subscribe(res => {
        if (res?.error) {
          setIsLoading(false)
          setError(res.error)
          unsubscribe()
          return
        }

        if (res?.data?.createThreadRun?.threadAssistantMessageCompleted) {
          stop.current()
        }

        setThreadRunItem(prevData =>
          combineThreadRunData(prevData, res.data?.createThreadRun)
        )
      })

    return unsubscribe
  }

  const deleteThreadMessagePair = useMutation(DeleteThreadMessagePairMutation)

  const sendUserMessage = (
    userMessage: CreateMessageInput,
    options?: ThreadRunOptionsInput
  ) => {
    if (isLoading) return

    setIsLoading(true)
    setError(undefined)
    setThreadRunItem(undefined)

    if (threadId) {
      unsubscribeFn.current = createThreadRun(userMessage, options)
    } else {
      unsubscribeFn.current = createThreadAndRun(userMessage, options)
    }
  }

  const onDeleteThreadMessagePair = (
    threadId: string,
    userMessageId: string,
    assistantMessageId: string
  ) => {
    return deleteThreadMessagePair({
      threadId,
      userMessageId,
      assistantMessageId
    })
      .then(res => {
        if (res?.data?.deleteThreadMessagePair) {
          return true
        }
        return false
      })
      .catch(e => {
        return false
      })
  }

  const regenerate = (payload: {
    threadId: string
    userMessageId: string
    assistantMessageId: string
    userMessage: CreateMessageInput
    threadRunOptions?: ThreadRunOptionsInput
  }) => {
    if (!threadId) return

    setIsLoading(true)
    setError(undefined)
    // 1. delete message pair
    deleteThreadMessagePair({
      threadId: payload.threadId,
      userMessageId: payload.userMessageId,
      assistantMessageId: payload.assistantMessageId
    })
      .then(res => {
        // 2. send userMessage
        if (res?.data?.deleteThreadMessagePair) {
          sendUserMessage(payload.userMessage, payload.threadRunOptions)
        }
      })
      .catch(e => {
        // FIXME error handling
      })
  }

  return {
    isLoading,
    answer: threadRunItem,
    error,
    setError,
    sendUserMessage,
    stop: stop.current,
    regenerate,
    deleteThreadMessagePair: onDeleteThreadMessagePair
  }
}
