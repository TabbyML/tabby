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

function parseVscodeNotebookCellURI(uri: string) {
  if (!uri.startsWith(VSCODE_NOTEBOOK_CELL_SCHEME)) return undefined

  const _lengths = ['W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f']
  const _padRegexp = new RegExp(`^[${_lengths.join('')}]+`)
  const _radix = 7
  const fragment = uri.split('#').pop() || ''
  const idx = fragment.indexOf('s')
  if (idx < 0) {
    return undefined
  }
  const handle = parseInt(
    fragment.substring(0, idx).replace(_padRegexp, ''),
    _radix
  )

  if (isNaN(handle)) {
    return undefined
  }
  return {
    handle
  }
}

export function resolveFileNameForDisplay(uri: string) {
  const regexPattern = `(?:${VSCODE_NOTEBOOK_CELL_SCHEME}:)?(.*?)(\\?|#|$)`
  const regex = new RegExp(regexPattern)

  const pathSegments = uri.split('/')
  const fileName = pathSegments[pathSegments.length - 1]
  const match = fileName.match(regex)
  const displayName = match ? match[1] : null
  const notebook = parseVscodeNotebookCellURI(uri)

  if (displayName && notebook) {
    return `${displayName} · Cell ${(notebook.handle || 0) + 1}`
  }
  return displayName
}
