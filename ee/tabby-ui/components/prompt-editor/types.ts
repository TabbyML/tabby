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
}

export type OptionItem = CategoryOptionItem | SourceOptionItem
