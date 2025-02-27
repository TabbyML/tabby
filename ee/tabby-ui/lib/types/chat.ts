import { Content } from '@tiptap/core'
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
   * The path to the file.
   *
   * This can be:
   * - A URI, e.g., `file:///path/to/file.txt` or `untitled://Untitled-1`
   * - A relative path, e.g., `path/to/file.txt`. In this case, either `baseDir` or `gitUrl` is required.
   */
  filepath: string

  /**
   * The client local URI to the base directory.
   *
   * This is provided when the file is contained in a local workspace that does not have a `gitUrl` available.
   * Example: `file:///path/to/file.txt`
   */
  baseDir?: string

  /**
   * The range of the selected content in the file.
   * If the range is not provided, the whole file is considered.
   */
  range?: { start: number; end: number }
  content: string

  /**
   * The URL of the git repository.
   *
   * This is provided when the file is contained in a git repository.
   * Example: `https://github.com/TabbyML/tabby`
   */
  gitUrl?: string
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

/**
 * The state used to restore the chat panel, should be json serializable.
 * Save this state to client so that the chat panel can be restored across webview reloading.
 */
export interface SessionState {
  threadId?: string | undefined
  qaPairs?: QuestionAnswerPair[] | undefined
  input?: Content | undefined
  relevantContext?: Context[] | undefined
  selectedRepoId?: string | undefined
}
