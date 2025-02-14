import React from 'react'

import { graphql } from '@/lib/gql/generates'

import {
  CreateMessageInput,
  CreateThreadRunSubscription as CreateThreadRunSubscriptionResponse,
  ThreadAssistantMessageReadingCode,
  ThreadRunOptionsInput
} from '../gql/generates/graphql'
import { client, useMutation } from '../tabby/gql'
import {
  ExtendedCombinedError,
  ThreadAssistantMessageAttachmentCodeHits,
  ThreadAssistantMessageAttachmentDocHits
} from '../types'
import { useLatest } from './use-latest'

interface UseThreadRunOptions {
  onError?: (err: Error) => void
  threadId?: string
  onAssistantMessageCompleted?: (answer: AnswerStream) => void
}

const CreateThreadAndRunSubscription = graphql(/* GraphQL */ `
  subscription CreateThreadAndRun($input: CreateThreadAndRunInput!) {
    createThreadAndRun(input: $input) {
      __typename
      ... on ThreadCreated {
        id
      }
      ... on ThreadUserMessageCreated {
        id
      }
      ... on ThreadAssistantMessageCreated {
        id
      }
      ... on ThreadAssistantMessageReadingCode {
        snippet
        fileList
      }
      ... on ThreadRelevantQuestions {
        questions
      }
      ... on ThreadAssistantMessageAttachmentsCodeFileList {
        codeFileList: fileList
      }
      ... on ThreadAssistantMessageAttachmentsCode {
        hits {
          code {
            gitUrl
            commit
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
      }
      ... on ThreadAssistantMessageAttachmentsDoc {
        hits {
          doc {
            __typename
            ... on MessageAttachmentWebDoc {
              title
              link
              content
            }
            ... on MessageAttachmentIssueDoc {
              title
              link
              author {
                id
                email
                name
              }
              body
              closed
            }
            ... on MessageAttachmentPullDoc {
              title
              link
              author {
                id
                email
                name
              }
              body
              merged
            }
          }
          score
        }
      }
      ... on ThreadAssistantMessageContentDelta {
        delta
      }
      ... on ThreadAssistantMessageCompleted {
        id
      }
    }
  }
`)

const CreateThreadRunSubscription = graphql(/* GraphQL */ `
  subscription CreateThreadRun($input: CreateThreadRunInput!) {
    createThreadRun(input: $input) {
      __typename
      ... on ThreadCreated {
        id
      }
      ... on ThreadUserMessageCreated {
        id
      }
      ... on ThreadAssistantMessageCreated {
        id
      }
      ... on ThreadAssistantMessageReadingCode {
        snippet
        fileList
      }
      ... on ThreadRelevantQuestions {
        questions
      }
      ... on ThreadAssistantMessageAttachmentsCodeFileList {
        codeFileList: fileList
      }
      ... on ThreadAssistantMessageAttachmentsCode {
        hits {
          code {
            gitUrl
            commit
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
      }
      ... on ThreadAssistantMessageAttachmentsDoc {
        hits {
          doc {
            __typename
            ... on MessageAttachmentWebDoc {
              title
              link
              content
            }
            ... on MessageAttachmentIssueDoc {
              title
              link
              author {
                id
                email
                name
              }
              body
              closed
            }
            ... on MessageAttachmentPullDoc {
              title
              link
              author {
                id
                email
                name
              }
              body
              merged
            }
          }
          score
        }
      }
      ... on ThreadAssistantMessageContentDelta {
        delta
      }
      ... on ThreadAssistantMessageCompleted {
        id
      }
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
export interface AnswerStream {
  threadId?: string
  userMessageId?: string
  assistantMessageId?: string
  codeSourceId?: string
  relevantQuestions?: Array<string>
  attachmentsCode?: ThreadAssistantMessageAttachmentCodeHits
  attachmentsDoc?: ThreadAssistantMessageAttachmentDocHits
  readingCode?: ThreadAssistantMessageReadingCode
  content: string
  isReadingCode: boolean
  isReadingFileList: boolean
  isReadingDocs: boolean
  completed: boolean
}

const defaultAnswerStream = (): AnswerStream => ({
  content: '',
  completed: false,
  isReadingCode: false,
  isReadingFileList: false,
  isReadingDocs: true
})

export interface ThreadRun {
  answer: AnswerStream

  isLoading: boolean

  error: ExtendedCombinedError | undefined

  sendUserMessage: (
    message: CreateMessageInput,
    options?: ThreadRunOptionsInput
  ) => void

  stop: (silent?: boolean) => void

  // if deletion fails, an error message will be returned
  regenerate: (payload: {
    threadId: string
    userMessageId: string
    assistantMessageId: string
    userMessage: CreateMessageInput
    threadRunOptions?: ThreadRunOptionsInput
  }) => Promise<string | void>

  // if deletion fails, an error message will be returned
  deleteThreadMessagePair: (
    threadId: string,
    userMessageId: string,
    assistantMessageId: string
  ) => Promise<string | void>
}

export function useThreadRun({
  threadId: propsThreadId,
  onAssistantMessageCompleted
}: UseThreadRunOptions): ThreadRun {
  const [threadId, setThreadId] = React.useState<string | undefined>(
    propsThreadId
  )
  const unsubscribeFn = React.useRef<(() => void) | undefined>()
  const [isLoading, setIsLoading] = React.useState(false)
  const [answerStream, setAnswerStream] = React.useState<AnswerStream>(
    defaultAnswerStream()
  )
  const [error, setError] = React.useState<ExtendedCombinedError | undefined>()
  const combineAnswerStream = (
    existingData: AnswerStream,
    data: CreateThreadRunSubscriptionResponse['createThreadRun']
  ): AnswerStream => {
    const x: AnswerStream = {
      ...existingData
    }

    switch (data.__typename) {
      case 'ThreadCreated':
        x.threadId = data.id
        break
      case 'ThreadUserMessageCreated':
        x.userMessageId = data.id
        break
      case 'ThreadAssistantMessageCreated':
        x.assistantMessageId = data.id
        break
      case 'ThreadRelevantQuestions':
        x.relevantQuestions = data.questions
        break
      case 'ThreadAssistantMessageReadingCode':
        x.isReadingCode = true
        x.isReadingFileList = true
        x.isReadingDocs = true
        x.readingCode = {
          fileList: data.fileList,
          snippet: data.snippet
        }
        break
      case 'ThreadAssistantMessageAttachmentsCodeFileList':
        x.isReadingFileList = false
        break
      case 'ThreadAssistantMessageAttachmentsCode':
        x.isReadingCode = false
        x.attachmentsCode = data.hits
        break
      case 'ThreadAssistantMessageAttachmentsDoc':
        x.attachmentsDoc = data.hits
        x.isReadingDocs = false
        break
      case 'ThreadAssistantMessageContentDelta':
        x.isReadingCode = false
        x.isReadingFileList = false
        x.isReadingDocs = false
        x.content += data.delta
        break
      case 'ThreadAssistantMessageCompleted':
        x.completed = true
        break
      default:
        // Ignore unknown event type.
        break
    }

    return x
  }

  const stop = useLatest((silent?: boolean) => {
    unsubscribeFn.current?.()
    unsubscribeFn.current = undefined
    setIsLoading(false)
    setAnswerStream(p => ({
      ...p,
      isReadingCode: false,
      isReadingFileList: false,
      isReadingDocs: false,
      completed: true
    }))

    if (!silent && threadId) {
      onAssistantMessageCompleted?.(answerStream)
    }
  })

  React.useEffect(() => {
    if (propsThreadId !== threadId) {
      setThreadId(propsThreadId)
    }
  }, [propsThreadId])

  const createThreadAndRun = (
    userMessage: CreateMessageInput,
    options?: ThreadRunOptionsInput
  ) => {
    const { unsubscribe } = client
      .subscription(CreateThreadAndRunSubscription, {
        input: {
          thread: {
            userMessage
          },
          options
        }
      })
      .subscribe(res => {
        if (res?.error) {
          setIsLoading(false)
          setError(res.error)
          unsubscribe()
          return
        }

        const value = res.data?.createThreadAndRun
        if (!value) {
          return
        }

        if (value?.__typename === 'ThreadAssistantMessageCompleted') {
          stop.current()
        }

        if (value?.__typename === 'ThreadCreated') {
          if (value.id !== threadId) {
            setThreadId(value.id)
          }
        }

        setAnswerStream(prevData => combineAnswerStream(prevData, value))
      })

    return unsubscribe
  }

  const createThreadRun = (
    userMessage: CreateMessageInput,
    options?: ThreadRunOptionsInput
  ) => {
    if (!threadId) return
    const { unsubscribe } = client
      .subscription(CreateThreadRunSubscription, {
        input: {
          threadId,
          additionalUserMessage: userMessage,
          options
        }
      })
      .subscribe(res => {
        if (res?.error) {
          setIsLoading(false)
          setError(res.error)
          unsubscribe()
          return
        }

        const value = res.data?.createThreadRun
        if (!value) {
          return
        }

        if (value.__typename === 'ThreadAssistantMessageCompleted') {
          stop.current()
        }

        setAnswerStream(prevData => combineAnswerStream(prevData, value))
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
    setAnswerStream(defaultAnswerStream())

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
  ): Promise<string | void> => {
    return deleteThreadMessagePair({
      threadId,
      userMessageId,
      assistantMessageId
    }).then(res => {
      if (!res?.data?.deleteThreadMessagePair) {
        if (res?.error) {
          throw res.error
        }

        throw new Error('Failed to fetch')
      }
    })
  }

  const regenerate = (payload: {
    threadId: string
    userMessageId: string
    assistantMessageId: string
    userMessage: CreateMessageInput
    threadRunOptions?: ThreadRunOptionsInput
  }) => {
    if (!threadId) return Promise.resolve(undefined)

    setIsLoading(true)
    // reset assistantMessage
    setError(undefined)
    setAnswerStream(defaultAnswerStream())

    // 1. delete message pair
    return onDeleteThreadMessagePair(
      payload.threadId,
      payload.userMessageId,
      payload.assistantMessageId
    )
      .then(() => {
        // 2. send a new user message
        sendUserMessage(payload.userMessage, payload.threadRunOptions)
      })
      .catch(e => {
        const error = e instanceof Error ? e : new Error('Failed to fetch')
        setError(error)
        setIsLoading(false)
      })
  }

  return {
    isLoading,
    answer: answerStream,
    error,
    sendUserMessage,
    stop: stop.current,
    regenerate,
    deleteThreadMessagePair: onDeleteThreadMessagePair
  }
}
