interface LineRange {
  start: number
  end: number
}

export interface FileContext {
  kind: 'file'
  range: LineRange
  filePath: string
  link: string
  language?: string
  // FIXME(jueliang): add code snippet here for client side mock
  content: string
}

export type Context = FileContext

export interface UserMessage {
  id: string
  message: string
  selectContext?: Context
  relevantContext?: Array<Context>
}

export type UserMessageWithOptionalId = Omit<UserMessage, 'id'> & {
  id?: string
}

export interface AssistantMessage {
  id: string
  message: string
  error?: string
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
