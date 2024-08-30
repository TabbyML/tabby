import { ContextInfo, ContextSource } from '@/lib/gql/generates/graphql'

export type CategoryOptionItem = {
  type: 'category'
  label: string
  kind: 'doc' | 'code'
}

export type SourceOptionItem = {
  type: 'source'
  label: string
  id: string
  kind: 'doc' | 'code'
  data: ContextSource
}

export type OptionItem = CategoryOptionItem | SourceOptionItem
