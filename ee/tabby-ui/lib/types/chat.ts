import type { components } from 'tabby-openapi'

import {
  ContextSourceKind,
  CreateThreadRunSubscription,
  MessageAttachmentCode,
  MessageCodeSearchHit,
  MessageDocSearchHit
} from '../gql/generates/graphql'
import { ArrayElementType } from './common'

export interface FileContext {
  kind: 'file'
  /**
   * filepath can be:
   * - uri, file://path/to/file.txt or untitled://Untitled-1
   * - relative path, path/to/file.txt, in this case, `baseDir` or `git_uri` is required
   */
  filepath: string
  /**
   * Uri to the base dir, provided when filepath is relative path and git_url is not available.
   */
  baseDir?: string
  /**
   * The range of the selected content in the file.
   * If the range is not provided, the whole file is considered.
   */
  range?: { start: number; end: number }
  content: string
  git_url: string
  commit?: string
}

export type Context = FileContext

export interface UserMessage {
  id: string
  message: string

  // Client side context - displayed in user message
  selectContext?: Context

  // Client side contexts - displayed in assistant message
  relevantContext?: Array<Context>

  // Client side active selection context - displayed in assistant message
  activeContext?: Context
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
export type AttachmentCodeItem =
  ArrayElementType<ThreadAssistantMessageAttachmentCodeHits>['code'] & {
    isClient?: boolean
    extra?: { scores?: MessageCodeSearchHit['scores'] }
    baseDir?: string
  }

// for rendering, including score
export type AttachmentDocItem =
  ArrayElementType<ThreadAssistantMessageAttachmentDocHits>['doc'] & {
    extra?: { score?: MessageDocSearchHit['score'] }
  }

export type MentionAttributes = {
  id: string
  label: string
  kind: ContextSourceKind
}
