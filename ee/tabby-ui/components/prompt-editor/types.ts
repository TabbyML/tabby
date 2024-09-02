import { ContextSource } from '@/lib/gql/generates/graphql'

export type CategoryOptionItem = {
  type: 'category'
  label: string
  category: 'doc' | 'code'
}

export type SourceOptionItem = {
  type: 'source'
  label: string
  id: string
  category: 'doc' | 'code'
  data: ContextSource
}

export type OptionItem = CategoryOptionItem | SourceOptionItem
