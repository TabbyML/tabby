import { uniq } from 'lodash-es'

import { ContextInfo, ContextKind } from '../gql/generates/graphql'
import { MentionAttributes } from '../types'

export const isCodeSourceContext = (kind: ContextKind) => {
  return [ContextKind.Git, ContextKind.Github, ContextKind.Gitlab].includes(
    kind
  )
}

export const isDocSourceContext = (kind: ContextKind) => {
  return [ContextKind.Doc, ContextKind.Web].includes(kind)
}

export const getMentionsFromText = (
  text: string,
  sources: ContextInfo['sources'] | undefined
) => {
  if (!sources?.length) return []

  const mentions: MentionAttributes[] = []
  const regex = /\[\[source:(\S+)\]\]/g
  let match
  while ((match = regex.exec(text))) {
    const sourceId = match[1]
    const source = sources?.find(o => o.sourceId === sourceId)
    if (source) {
      mentions.push({
        id: sourceId,
        label: source.displayName,
        kind: source.kind
      })
    }
  }
  return mentions
}

export const getSourceIdsFromMentions = (mentions: MentionAttributes[]) => {
  const docSourceIds: string[] = []
  const codeSourceIds: string[] = []
  let searchPublic = false
  for (let mention of mentions) {
    const { kind, id } = mention
    if (isCodeSourceContext(kind)) {
      codeSourceIds.push(id)
    } else if (kind === ContextKind.Web) {
      searchPublic = true
    } else {
      docSourceIds.push(id)
    }
  }

  return {
    searchPublic,
    docSourceIds: uniq(docSourceIds),
    codeSourceIds: uniq(codeSourceIds)
  }
}
