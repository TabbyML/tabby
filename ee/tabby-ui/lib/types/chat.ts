import { type Message } from 'ai'

export interface Chat extends Record<string, any> {
  id: string
  title: string
  createdAt: Date
  userId: string
  path: string
  messages: Message[]
  sharePath?: string
}

export type ISearchHit = {
  id: number
  score: number
  doc: {
    body: string
    filepath: string
    git_url: string
    language: string
    start_line: number
  }
}
export type SearchReponse = {
  hits: Array<ISearchHit>
  num_hits: number
}

export type MessageActionType = 'edit' | 'delete' | 'regenerate'
