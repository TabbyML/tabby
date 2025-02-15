import { clsx, type ClassValue } from 'clsx'
import { compact, isNil } from 'lodash-es'
import { customAlphabet } from 'nanoid'
import type {
  ChatCommand,
  EditorContext,
  FileLocation,
  Filepath,
  LineRange,
  Location,
  Position,
  PositionRange
} from 'tabby-chat-panel'
import { twMerge } from 'tailwind-merge'

import { AttachmentCodeItem, AttachmentDocItem, FileContext } from '@/lib/types'

import { Maybe } from '../gql/generates/graphql'

export * from './chat'
export * from './repository'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export const nanoid = customAlphabet(
  '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',
  7
) // 7-character random string

export async function fetcher<JSON = any>(
  input: RequestInfo,
  init?: RequestInit
): Promise<JSON> {
  const res = await fetch(input, init)

  if (!res.ok) {
    const json = await res.json()
    if (json.error) {
      const error = new Error(json.error) as Error & {
        status: number
      }
      error.status = res.status
      throw error
    } else {
      throw new Error('An unexpected error occurred')
    }
  }

  return res.json()
}

export function formatDate(input: string | number | Date): string {
  const date = new Date(input)
  return date.toLocaleDateString('en-US', {
    month: 'long',
    day: 'numeric',
    year: 'numeric'
  })
}

export function truncateText(
  text: string,
  maxLength = 50,
  delimiters = /[ ,.:;\n，。：；]/
) {
  if (!text) return ''
  if (text.length <= maxLength) {
    return text
  }

  let truncatedText = text.slice(0, maxLength)

  let lastDelimiterIndex = -1
  for (let i = maxLength - 1; i >= 0; i--) {
    if (delimiters.test(truncatedText[i])) {
      lastDelimiterIndex = i
      break
    }
  }

  if (lastDelimiterIndex !== -1) {
    truncatedText = truncatedText.slice(0, lastDelimiterIndex)
  }

  return truncatedText + '...'
}

export const isClientSide = () => {
  return typeof window !== 'undefined'
}

export const delay = (ms: number) => {
  return new Promise(resolve => {
    setTimeout(() => resolve(null), ms)
  })
}

export function formatLineHashForCodeBrowser(
  range:
    | {
        start: number
        end?: number
      }
    | undefined
): string {
  if (!range) return ''

  const { start, end } = range
  if (isNil(start) || isNaN(start)) return ''
  if (start === end) return `L${start}`
  return compact(
    [start, end].map(num =>
      typeof num === 'number' && !isNaN(num) ? `L${num}` : undefined
    )
  ).join('-')
}

export function formatLineHashForLocation(location: Location | undefined) {
  if (!location) {
    return ''
  }
  if (typeof location === 'number') {
    return `L${location}`
  }
  if (
    typeof location === 'object' &&
    'line' in location &&
    typeof location.line === 'number'
  ) {
    return `L${location.line}`
  }
  if ('start' in location) {
    const start = location.start
    if (typeof start === 'number') {
      const end = location.end as number
      return `L${start}-L${end}`
    }
    if (
      typeof start === 'object' &&
      'line' in start &&
      typeof start.line === 'number'
    ) {
      const end = location.end as Position
      return `L${start.line}-L${end.line}`
    }
  }
  return ''
}

export function getRangeFromAttachmentCode(code: {
  startLine?: Maybe<number>
  content: string
}): LineRange | undefined {
  if (!code?.startLine) return undefined

  const start = code.startLine
  const lineCount = code.content.split('\n').length
  const end = start + lineCount - 1

  return {
    start,
    end
  }
}

export function getRangeTextFromAttachmentCode(code: AttachmentCodeItem) {
  const range = getRangeFromAttachmentCode(code)
  return formatLineHashForCodeBrowser(range)
}

export function getContent(item: AttachmentDocItem) {
  switch (item.__typename) {
    case 'MessageAttachmentWebDoc':
      return item.content
    case 'MessageAttachmentIssueDoc':
    case 'MessageAttachmentPullDoc':
      return item.body
  }

  return ''
}

export function getPromptForChatCommand(command: ChatCommand) {
  switch (command) {
    case 'explain':
      return 'Explain the selected code:'
    case 'fix':
      return 'Identify and fix potential bugs in the selected code:'
    case 'generate-docs':
      return 'Generate documentation for the selected code:'
    case 'generate-tests':
      return 'Generate a unit test for the selected code:'
  }
}

export const convertFilepath = (filepath: Filepath) => {
  if (filepath.kind === 'git') {
    return {
      filepath: filepath.filepath,
      git_url: filepath.gitUrl
    }
  }
  if (filepath.kind === 'workspace') {
    return {
      filepath: filepath.filepath,
      baseDir: filepath.baseDir,
      git_url: ''
    }
  }
  return {
    filepath: filepath.uri,
    git_url: ''
  }
}

export function convertEditorContext(
  editorContext: EditorContext
): FileContext {
  const convertRange = (range: LineRange | PositionRange | undefined) => {
    // If the range is not provided, the whole file is considered.
    if (!range || typeof range.start === 'undefined') {
      return undefined
    }

    if (typeof range.start === 'number') {
      return range as LineRange
    }

    const positionRange = range as PositionRange
    return {
      start: positionRange.start.line,
      end: positionRange.end.line
    }
  }

  return {
    kind: 'file',
    content: editorContext.content,
    range: convertRange(editorContext.range),
    ...convertFilepath(editorContext.filepath)
  }
}

export function getFilepathFromContext(context: FileContext): Filepath {
  if (context.git_url.length > 1 && !context.filepath.includes(':')) {
    return {
      kind: 'git',
      filepath: context.filepath,
      gitUrl: context.git_url
    }
  }
  if (
    context.baseDir &&
    context.baseDir.length > 1 &&
    !context.filepath.includes(':')
  ) {
    return {
      kind: 'workspace',
      filepath: context.filepath,
      baseDir: context.baseDir
    }
  }
  return {
    kind: 'uri',
    uri: context.filepath
  }
}

export function getFileLocationFromContext(context: FileContext): FileLocation {
  return {
    filepath: getFilepathFromContext(context),
    location: context.range
  }
}

export function buildCodeBrowserUrlForContext(
  base: string,
  context: FileContext
) {
  const url = new URL(base)
  url.pathname = '/files'

  const searchParams = new URLSearchParams()
  searchParams.append('redirect_filepath', context.filepath)
  searchParams.append('redirect_git_url', context.git_url)
  if (context.commit) {
    searchParams.append('redirect_rev', context.commit)
  }

  url.search = searchParams.toString()

  url.hash = formatLineHashForCodeBrowser(context.range)

  return url.toString()
}
