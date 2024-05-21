import { Message } from 'ai'
import type { ChatMessage } from 'tabby-chat-panel'

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
  relevant_code?: AnswerResponse['relevant_code']
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

export type CodeSearchDocument = {
  body: string
  filepath: string
  git_url: string
  language?: string
  start_line?: number
}

export type AnswerRequest = {
  messages: Array<Message>
  code_query?: {
    git_url: string
    filepath: string
    language: string
    content: string
  }
  doc_query?: boolean
  generate_relevant_questions?: boolean
}

export type AnswerResponse = {
  relevant_code?: Array<CodeSearchDocument>
  relevant_documents?: Array<CodeSearchDocument>
  relevant_questions?: Array<string>
  answer_delta?: string
}
