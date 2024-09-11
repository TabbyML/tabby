import { ContextSource } from '@/lib/gql/generates/graphql'

export type SourceOptionItem = {
  label: string
  id: string
  category: 'doc' | 'code'
  data: ContextSource
}

export type OptionItem = SourceOptionItem
