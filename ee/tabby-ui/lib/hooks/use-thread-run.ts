import React from 'react'
import { pickBy } from 'lodash-es'
import { OperationContext, useSubscription } from 'urql'

import { graphql } from '@/lib/gql/generates'

import {
  CreateMessageInput,
  ThreadRunItem,
  ThreadRunOptionsInput
} from '../gql/generates/graphql'
import { useDebounceCallback } from './use-debounce'

interface UseThreadRunOptions {
  onError?: (err: Error) => void
  threadId?: string
  headers?: Record<string, string> | Headers
  onThreadCreated?: (threadId: string) => void
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

export function useThreadRun({
  onError,
  threadId: propsThreadId,
  headers,
  onThreadCreated
}: UseThreadRunOptions) {
  const [threadId, setThreadId] = React.useState<string | undefined>(
    propsThreadId
  )

  const [pause, setPause] = React.useState(true)
  const [followupPause, setFollowupPause] = React.useState(true)
  const [createMessageInput, setCreateMessageInput] =
    React.useState<CreateMessageInput | null>(null)
  const [threadRunOptions, setThreadRunOptions] = React.useState<
    ThreadRunOptionsInput | undefined
  >()
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
    data: ThreadRunItem
  ): ThreadRunItem => {
    if (!data) return data
    // new thread created
    if (data.threadCreated) return data
    // new userMessage created
    if (
      existingData?.threadAssistantMessageCreated &&
      data.threadUserMessageCreated
    )
      return data

    return {
      ...existingData,
      ...pickBy(data, v => v !== null),
      threadAssistantMessageContentDelta: `${
        existingData?.threadAssistantMessageContentDelta ?? ''
      }${data?.threadAssistantMessageContentDelta ?? ''}`
    }
  }

  const debouncedStop = useDebounceCallback(
    (silent?: boolean) => {
      setPause(true)
      setFollowupPause(true)
      setIsLoading(false)
      if (!silent && !propsThreadId && threadId) {
        onThreadCreated?.(threadId)
      }
    },
    300,
    {
      leading: false,
      onUnmount(debounced) {
        if (isLoading) {
          debounced(true)
        }
        debounced.flush()
      }
    }
  )

  const stop = (silent?: boolean) => debouncedStop.run(silent)

  const [createThreadAndRunResult] = useSubscription(
    {
      query: CreateThreadAndRunSubscription,
      pause,
      variables: {
        input: {
          thread: {
            userMessage: createMessageInput as CreateMessageInput
          },
          options: threadRunOptions
        }
      },
      context: operationContext
    },
    (prevData, data) => {
      return {
        ...data,
        createThreadAndRun: combineThreadRunData(
          prevData?.createThreadAndRun,
          data.createThreadAndRun
        )
      }
    }
  )

  const [createThreadRunResult] = useSubscription(
    {
      query: CreateThreadRunSubscription,
      pause: followupPause,
      variables: {
        input: {
          threadId: threadId as string,
          additionalUserMessage: createMessageInput as CreateMessageInput,
          options: threadRunOptions
        }
      },
      context: operationContext
    },
    (prevData, data) => {
      return {
        ...data,
        createThreadRun: combineThreadRunData(
          prevData?.createThreadRun,
          data.createThreadRun
        )
      }
    }
  )

  React.useEffect(() => {
    if (propsThreadId && propsThreadId !== threadId) {
      setThreadId(propsThreadId)
    }
  }, [propsThreadId])

  // createThreadAndRun
  React.useEffect(() => {
    if (
      createThreadAndRunResult?.data?.createThreadAndRun
        ?.threadAssistantMessageCompleted
    ) {
      stop()
    }
    if (
      createThreadAndRunResult.fetching ||
      !createThreadAndRunResult?.operation
    )
      return
    // error handling
    if (createThreadAndRunResult?.error) {
      setError(createThreadAndRunResult?.error)
      stop()
      return
    }
    // save the threadId
    if (createThreadAndRunResult?.data?.createThreadAndRun?.threadCreated) {
      setThreadId(
        createThreadAndRunResult?.data?.createThreadAndRun?.threadCreated
      )
    }
    if (createThreadAndRunResult?.data?.createThreadAndRun) {
      setThreadRunItem(createThreadAndRunResult?.data?.createThreadAndRun)
    }
  }, [createThreadAndRunResult])

  // createThreadRun
  React.useEffect(() => {
    if (
      createThreadRunResult?.data?.createThreadRun
        ?.threadAssistantMessageCompleted
    ) {
      stop()
    }

    if (createThreadRunResult?.fetching || !createThreadRunResult?.operation)
      return

    // error handling
    if (createThreadRunResult?.error) {
      setError(createThreadRunResult?.error)
      stop()
      return
    }

    if (createThreadRunResult?.data?.createThreadRun) {
      setThreadRunItem(createThreadRunResult?.data?.createThreadRun)
    }
  }, [createThreadRunResult])

  const triggerRequest = async (
    userMessage: CreateMessageInput,
    options?: ThreadRunOptionsInput
  ) => {
    setIsLoading(true)
    setError(undefined)
    setThreadRunItem(undefined)

    setCreateMessageInput(userMessage)
    setThreadRunOptions(options)

    if (threadId) {
      setFollowupPause(false)
    } else {
      setPause(false)
    }
  }

  return {
    isLoading,
    answer: threadRunItem,
    error,
    setError,
    triggerRequest,
    stop
  }
}
