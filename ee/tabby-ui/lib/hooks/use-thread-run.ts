import React, { useEffect } from 'react'
import { isEmpty, pickBy } from 'lodash-es'
import { createRequest, OperationContext, useSubscription } from 'urql'

import { graphql } from '@/lib/gql/generates'

import {
  CreateMessageInput,
  ThreadRunItem,
  ThreadRunOptionsInput
} from '../gql/generates/graphql'
import { client } from '../tabby/gql'
import { useLatest } from './use-latest'

interface UseThreadRunOptions {
  onError?: (err: Error) => void
  threadId?: string
  headers?: Record<string, string> | Headers
  threadRunOptions?: ThreadRunOptionsInput
}

const CreateThreadAndRunSubscription = graphql(/* GraphQL */ `
  subscription CreateThreadAndRun($input: CreateThreadAndRunInput!) {
    createThreadAndRun(input: $input) {
      threadCreated
      threadRelevantQuestions
      threadUserMessageCreated
      threadAssistantMessageCreated
      threadAssistantMessageAttachmentsCode {
        filepath
        content
      }
      threadAssistantMessageAttachmentsDoc {
        title
        link
        content
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
      threadRelevantQuestions
      threadUserMessageCreated
      threadAssistantMessageCreated
      threadAssistantMessageAttachmentsCode {
        filepath
        content
      }
      threadAssistantMessageAttachmentsDoc {
        title
        link
        content
      }
      threadAssistantMessageContentDelta
      threadAssistantMessageCompleted
    }
  }
`)

export function useThreadRun({
  onError,
  threadId: propsThreadId,
  threadRunOptions,
  headers
}: UseThreadRunOptions) {
  const [threadId, setThreadId] = React.useState<string | undefined>(
    propsThreadId
  )
  // FIXME use two pause
  const [pause, setPause] = React.useState(true)
  const [followupPause, setFollowupPause] = React.useState(true)
  const [createMessageInput, setCreateMessageInput] =
    React.useState<CreateMessageInput | null>(null)
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

  const [createThreadAndRunResult] = useSubscription({
    query: CreateThreadAndRunSubscription,
    pause: !createMessageInput ? true : pause,
    variables: {
      input: {
        thread: {
          userMessage: createMessageInput as CreateMessageInput
        },
        options: {
          ...threadRunOptions,
          docQuery: {
            content: createMessageInput?.content ?? ''
          }
        }
      }
    },
    context: operationContext
  })

  const [createThreadRunResult] = useSubscription({
    query: CreateThreadRunSubscription,
    pause: threadId && createMessageInput ? followupPause : true,
    variables: {
      input: {
        threadId: threadId as string,
        additionalUserMessage: createMessageInput as CreateMessageInput,
        options: threadRunOptions
      }
    },
    context: operationContext
  })

  useEffect(() => {
    // error handling
    if (createThreadAndRunResult?.error) {
      setError(createThreadAndRunResult?.error)
      return
    }
    // save the threadId
    if (createThreadAndRunResult?.data?.createThreadAndRun?.threadCreated) {
      setThreadId(
        createThreadAndRunResult?.data?.createThreadAndRun?.threadCreated
      )
    }
    if (createThreadAndRunResult?.data?.createThreadAndRun) {
      setThreadRunItem(prev =>
        mergeParsedThreadData(
          prev,
          createThreadAndRunResult?.data?.createThreadAndRun!
        )
      )
      if (
        createThreadAndRunResult?.data?.createThreadAndRun
          ?.threadAssistantMessageCompleted
      ) {
        setIsLoading(false)
        setPause(true)
        setFollowupPause(true)
      }
    }
  }, [createThreadAndRunResult])

  useEffect(() => {
    // error handling
    if (createThreadRunResult?.error) {
      setError(createThreadRunResult?.error)
      return
    }

    if (createThreadRunResult?.data?.createThreadRun) {
      setThreadRunItem(prev =>
        mergeParsedThreadData(
          prev,
          createThreadRunResult?.data?.createThreadRun!
        )
      )
      if (
        createThreadRunResult?.data?.createThreadRun
          ?.threadAssistantMessageCompleted
      ) {
        setIsLoading(false)
      }
    }
  }, [createThreadRunResult])

  const stop = () => {
    setPause(true)
    setFollowupPause(true)
    setIsLoading(false)
  }

  const mergeParsedThreadData = (
    existingData: ThreadRunItem | undefined,
    data: ThreadRunItem
  ): ThreadRunItem => {
    if (!existingData) return data

    return {
      ...existingData,
      ...pickBy(data, v => !isEmpty(v)),
      threadAssistantMessageContentDelta: `${
        existingData?.threadAssistantMessageContentDelta ?? ''
      }${data?.threadAssistantMessageContentDelta ?? ''}`
    }
  }

  // FIXME triggerWithOptions
  const triggerRequest = async (userMessage: CreateMessageInput) => {
    setIsLoading(true)
    setError(undefined)
    setThreadRunItem(undefined)

    setCreateMessageInput(userMessage)
    // if it's more convient to create a operation so that we can use the lastest token to connect websocket
    // const operation = client.createRequestOperation('subscription', createRequest(CreateThreadAndRunSubscription, {
    //   input: {
    //     thread: {
    //       userMessage
    //     },
    //     options: {
    //       generateRelevantQuestions: true
    //     }
    //   }
    // }))

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
