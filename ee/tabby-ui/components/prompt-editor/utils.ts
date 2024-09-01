import { Editor, JSONContent } from '@tiptap/react'

import { ContextKind } from '@/lib/gql/generates/graphql'
import { MentionAttributes } from '@/lib/types'

export const isRepositorySource = (kind: ContextKind) => {
  return [ContextKind.Git, ContextKind.Github, ContextKind.Gitlab].includes(
    kind
  )
}

export const getMentionsWithIndices = (editor: Editor) => {
  const json = editor.getJSON()
  const mentions: MentionAttributes[] = []
  let textLength = 0

  const traverse = (node: JSONContent) => {
    if (node.type === 'text') {
      textLength += node?.text?.length || 0
    } else if (node.type === 'mention') {
      if (node?.attrs?.id) {
        mentions.push({
          id: node.attrs.id,
          label: node.attrs.label,
          kind: node.attrs.kind
        })
      }
    }

    if (node.content) {
      node.content.forEach(traverse)
    }
  }

  traverse(json)
  return mentions
}
