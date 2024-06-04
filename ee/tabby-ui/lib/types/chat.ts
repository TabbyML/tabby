import type { ChatMessage } from 'tabby-chat-panel'
import type { components as TabbyOpenApiComponents } from 'tabby-openapi'

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

export type AnswerRequest = TabbyOpenApiComponents['schemas']['AnswerRequest']

type AnswerResponseChunk =
  TabbyOpenApiComponents['schemas']['AnswerResponseChunk']

export type AnswerResponse = {
  relevant_code?: AnswerResponseChunk['relevant_code']
  relevant_documents?: AnswerResponseChunk['relevant_documents']
  relevant_questions?: AnswerResponseChunk['relevant_questions']
  answer_delta?: AnswerResponseChunk['answer_delta']
}
