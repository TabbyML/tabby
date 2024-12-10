import { uniq } from 'lodash-es'

import {
  ContextInfo,
  ContextSource,
  ContextSourceKind
} from '@/lib/gql/generates/graphql'
import { MentionAttributes } from '@/lib/types'

import { VSCODE_NOTEBOOK_CELL_SCHEME } from '../constants'
import { MARKDOWN_SOURCE_REGEX } from '../constants/regex'

export const isCodeSourceContext = (kind: ContextSourceKind) => {
  return [
    ContextSourceKind.Git,
    ContextSourceKind.Github,
    ContextSourceKind.Gitlab
  ].includes(kind)
}

export const isDocSourceContext = (kind: ContextSourceKind) => {
  return [ContextSourceKind.Doc, ContextSourceKind.Web].includes(kind)
}

export const getMentionsFromText = (
  text: string,
  sources: ContextInfo['sources'] | undefined
) => {
  if (!sources?.length) return []

  const mentions: MentionAttributes[] = []
  let match
  while ((match = MARKDOWN_SOURCE_REGEX.exec(text))) {
    const sourceId = match[1]
    const source = sources?.find(o => o.sourceId === sourceId)
    if (source) {
      mentions.push({
        id: sourceId,
        label: source.sourceName,
        kind: source.sourceKind
      })
    }
  }
  return mentions
}

export const getThreadRunContextsFromMentions = (
  mentions: MentionAttributes[]
) => {
  const docSourceIds: string[] = []
  const codeSourceIds: string[] = []
  let searchPublic = false
  for (let mention of mentions) {
    const { kind, id } = mention
    if (isCodeSourceContext(kind)) {
      codeSourceIds.push(id)
    } else if (kind === ContextSourceKind.Web) {
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

export function getTitleFromMessages(
  sources: ContextSource[],
  content: string,
  options?: { maxLength?: number }
) {
  const firstLine = content.split('\n')[0] ?? ''
  const cleanedLine = firstLine
    .replace(MARKDOWN_SOURCE_REGEX, value => {
      const sourceId = value.slice(9, -2)
      const source = sources.find(s => s.sourceId === sourceId)
      return source?.sourceName ?? ''
    })
    .trim()

  let title = cleanedLine
  if (options?.maxLength) {
    title = title.slice(0, options?.maxLength)
  }
  return title
}

export function checkSourcesAvailability(
  sources: ContextInfo['sources'] | undefined
) {
  let hasCodebaseSource = false
  let hasDocumentSource = false
  if (sources) {
    sources.forEach(source => {
      if (isCodeSourceContext(source.sourceKind)) {
        hasCodebaseSource = true
      } else if (isDocSourceContext(source.sourceKind)) {
        hasDocumentSource = true
      }
    })
  }

  return { hasCodebaseSource, hasDocumentSource }
}

export function formatFileNameForDisplay(uri: string) {
  const regexPattern = `(?:${VSCODE_NOTEBOOK_CELL_SCHEME}:)?(.*?)(\\?|#|$)`
  const regex = new RegExp(regexPattern)
  const match = uri.match(regex)
  return match ? match[1] : null
}
