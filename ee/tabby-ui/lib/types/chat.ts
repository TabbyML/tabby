import { Content } from '@tiptap/core'
import type { components } from 'tabby-openapi'

import {
  ContextSourceKind,
  CreatePageRunSubscription,
  CreateThreadRunSubscription,
  Maybe,
  MessageAttachmentClientCode,
  MessageAttachmentCodeFileList,
  MessageCodeSearchHit,
  MessageDocSearchHit,
  ThreadAssistantMessageReadingCode,
  ThreadAssistantMessageReadingDoc
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

interface TerminalContext {
  kind: 'terminal'

  /**
   * The terminal name.
   */
  name: string

  /**
   * The terminal process id
   */
  processId: number | undefined

  /**
   * The selected text in the terminal.
   */
  selection: string
}

export type Context = FileContext | TerminalContext

export interface UserMessage {
  id: string
  content: string

  // Client side context - displayed in user message, eg. explain code
  selectContext?: Context

  // Client side contexts - displayed in assistant message, eg. add selection to code
  relevantContext?: Array<Context>

  // Client side active selection context - displayed in assistant message
  activeContext?: Context

  // codeSourceId?: string
}

export type UserMessageWithOptionalId = Omit<UserMessage, 'id'> & {
  id?: string
}

export interface AssistantMessage {
  id: string
  content: string
  error?: string
  attachment?: {
    clientCode?: Maybe<Array<MessageAttachmentClientCode>> | undefined
    code: Maybe<Array<AttachmentCodeItem>> | undefined
    doc: Maybe<Array<AttachmentDocItem>> | undefined
    codeFileList?: Maybe<MessageAttachmentCodeFileList>
  }
  readingCode?: ThreadAssistantMessageReadingCode
  readingDoc?: ThreadAssistantMessageReadingDoc
  isReadingCode?: boolean
  isReadingFileList?: boolean
  isReadingDocs?: boolean
  codeSourceId?: Maybe<string>
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
    gitUrl?: string
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

export interface ThreadRunContexts {
  modelName?: string
  searchPublic?: boolean
  docSourceIds?: string[]
  codeSourceId?: string
}

export interface ServerFileContext extends FileContext {
  extra?: {
    scores?: components['schemas']['CodeSearchScores']
  }
}
export type RelevantCodeContext = Context | ServerFileContext

type ExtractHitsByType<T, N> = T extends {
  __typename: N
  hits: infer H
}
  ? H
  : never

type ExtractDocByType<T, N> = T extends {
  __typename: N
  doc: infer H
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

export type PageSectionAttachmentDocHits = ExtractDocByType<
  CreatePageRunSubscription['createPageRun'],
  'PageSectionAttachmentDoc'
>

// for rendering, including scores
export type AttachmentCodeItem = Omit<
  ArrayElementType<ThreadAssistantMessageAttachmentCodeHits>['code'],
  '__typename'
> & {
  isClient?: boolean
  extra?: { scores?: MessageCodeSearchHit['scores'] }
  baseDir?: string
  __typename?: 'MessageAttachmentCode' | 'AttachmentCode'
}

// for rendering, including score
export type AttachmentDocItem =
  | (ArrayElementType<ThreadAssistantMessageAttachmentDocHits>['doc'] & {
      extra?: { score?: MessageDocSearchHit['score'] }
    })
  | (ArrayElementType<PageSectionAttachmentDocHits>['doc'] & {
      extra?: { score?: MessageDocSearchHit['score'] }
    })

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
