import { uniq } from 'lodash-es'

import {
  ContextInfo,
  ContextSource,
  ContextSourceKind
} from '@/lib/gql/generates/graphql'
import { MentionAttributes } from '@/lib/types'

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

/**
 * url e.g #cell=1
 * @param fragment
 * @returns
 */
function parseNotebookCellUriFragment(fragment: string) {
  if (!fragment) return undefined
  try {
    const searchParams = new URLSearchParams(fragment)
    const cellString = searchParams.get('cell')?.toString()
    if (!cellString) {
      return undefined
    }

    const handle = parseInt(cellString, 10)

    if (isNaN(handle)) {
      return undefined
    }
    return {
      handle
    }
  } catch (error) {
    return undefined
  }
}

export function resolveFileNameForDisplay(uri: string) {
  let url: URL
  try {
    url = new URL(uri)
  } catch (e) {
    url = new URL(uri, 'file://')
  }
  const filename = url.pathname.split('/').pop() || ''
  const extname = filename.includes('.') ? `.${filename.split('.').pop()}` : ''
  const isNotebook = extname.startsWith('.ipynb')
  const hash = url.hash ? url.hash.substring(1) : ''
  const cell = parseNotebookCellUriFragment(hash)
  if (isNotebook && cell) {
    return `${filename} · Cell ${(cell.handle || 0) + 1}`
  }
  return filename
}
