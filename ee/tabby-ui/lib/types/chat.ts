import type { ChatMessage, Context } from 'tabby-chat-panel'
import type {
  components,
  components as TabbyOpenApiComponents
} from 'tabby-openapi'

import {
  MessageAttachmentCode,
  MessageAttachmentDoc,
  MessageCodeSearchHit,
  MessageDocSearchHit,
  Repository
} from '../gql/generates/graphql'

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

export type MessageActionType = 'delete' | 'regenerate'

export type AnswerRequest = TabbyOpenApiComponents['schemas']['AnswerRequest']

type Keys<T> = T extends any ? keyof T : never
type Pick<T, K extends Keys<T>> = T extends { [k in K]?: any }
  ? T[K]
  : undefined
type MergeUnionType<T> = {
  [k in Keys<T>]?: Pick<T, k>
}
export type AnswerResponse = MergeUnionType<
  TabbyOpenApiComponents['schemas']['AnswerResponseChunk']
>

export type AnswerEngineExtraContext = {
  repository?: Omit<Repository, 'refs'>
}

export interface RelevantCodeContext extends Context {
  extra?: {
    scores?: components['schemas']['CodeSearchScores']
  }
}

// for rendering, including scores
export type AttachmentCodeItem = MessageAttachmentCode & {
  extra?: { scores?: MessageCodeSearchHit['scores'] }
}
// for rendering, including score
export type AttachmentDocItem = MessageAttachmentDoc & {
  extra?: { score?: MessageDocSearchHit['score'] }
}
