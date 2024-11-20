import type { ChatMessage, Context } from 'tabby-chat-panel'
import type { components } from 'tabby-openapi'

import {
  ContextSourceKind,
  CreateThreadRunSubscription,
  MessageAttachmentCode,
  MessageCodeSearchHit,
  MessageDocSearchHit
} from '../gql/generates/graphql'
import { ArrayElementType } from './common'

export interface UserMessage extends ChatMessage {
  id: string
}

export type UserMessageWithOptionalId = Omit<UserMessage, 'id'> & {
  id?: string
}

export interface AssistantMessage {
  id: string
  message: string
  error?: string
  relevant_code?: MessageAttachmentCode[]
}

export interface QuestionAnswerPair {
  user: UserMessage
  assistant?: AssistantMessage
}

export interface Chat extends Record<string, any> {
  id: string
  title: string
  createdAt: Date
  userId: string
  path: string
  messages: QuestionAnswerPair[]
  sharePath?: string
}

export type ISearchHit = {
  id: number
  doc?: {
    body?: string
    name?: string
    filepath?: string
    git_url?: string
    kind?: string
    language?: string
  }
}

export type SearchReponse = {
  hits?: Array<ISearchHit>
  num_hits?: number
}

export type MessageActionType = 'delete' | 'regenerate' | 'edit'

type Keys<T> = T extends any ? keyof T : never
type Pick<T, K extends Keys<T>> = T extends { [k in K]?: any }
  ? T[K]
  : undefined
type MergeUnionType<T> = {
  [k in Keys<T>]?: Pick<T, k>
}

export type ThreadRunContexts = {
  modelName?: string
  searchPublic?: boolean
  docSourceIds?: string[]
  codeSourceIds?: string[]
}

export interface RelevantCodeContext extends Context {
  extra?: {
    scores?: components['schemas']['CodeSearchScores']
  }
}

type ExtractHitsByType<T, N> = T extends {
  __typename: N
  hits: infer H
}
  ? H
  : never

export type ThreadAssistantMessageAttachmentCodeHits = ExtractHitsByType<
  CreateThreadRunSubscription['createThreadRun'],
  'ThreadAssistantMessageAttachmentsCode'
>
export type ThreadAssistantMessageAttachmentDocHits = ExtractHitsByType<
  CreateThreadRunSubscription['createThreadRun'],
  'ThreadAssistantMessageAttachmentsDoc'
>

// for rendering, including scores
export type AttachmentCode =
  ArrayElementType<ThreadAssistantMessageAttachmentCodeHits>['code'] & {
    isClient?: boolean
    extra?: { scores?: MessageCodeSearchHit['scores'] }
  }
// for rendering, including score
export type AttachmentDoc =
  ArrayElementType<ThreadAssistantMessageAttachmentDocHits>['doc'] & {
    extra?: { score?: MessageDocSearchHit['score'] }
  }

export type MentionAttributes = {
  id: string
  label: string
  kind: ContextSourceKind
}
