import DOMPurify from 'dompurify'
import he from 'he'
import { uniq } from 'lodash-es'
import { marked } from 'marked'
import moment from 'moment'
import type {
  ChangeItem,
  Filepath,
  FileRange,
  GetChangesParams,
  TerminalContext
} from 'tabby-chat-panel'

import {
  ContextInfo,
  ContextSource,
  ContextSourceKind,
  MessageAttachmentCodeInput
} from '@/lib/gql/generates/graphql'
import type { MentionAttributes } from '@/lib/types'
import {
  convertChangeItemsToContextContent,
  convertContextBlockToPlaceholder,
  formatObjectToMarkdownBlock,
  shouldAddPrefixNewline,
  shouldAddSuffixNewline
} from '@/lib/utils/markdown'

import {
  MARKDOWN_FILE_REGEX,
  MARKDOWN_SOURCE_REGEX,
  PLACEHOLDER_COMMAND_REGEX,
  PLACEHOLDER_FILE_REGEX,
  PLACEHOLDER_SYMBOL_REGEX
} from '../constants/regex'

export const isCodeSourceContext = (kind: ContextSourceKind) => {
  return [
    ContextSourceKind.Git,
    ContextSourceKind.Github,
    ContextSourceKind.Gitlab
  ].includes(kind)
}

export const isDocSourceContext = (kind: ContextSourceKind) => {
  return [
    ContextSourceKind.Doc,
    ContextSourceKind.Web,
    ContextSourceKind.Ingested
  ].includes(kind)
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

/**
 * Get the file mention from the text
 * @param text
 * @returns {Array<{filepath: Filepath}>}
 */
export const getFileMentionFromText = (text: string) => {
  if (!text) return []

  const mentions: Array<{ filepath: Filepath }> = []
  let match
  while ((match = MARKDOWN_FILE_REGEX.exec(text))) {
    const fileItem = match[1]
    if (fileItem) {
      try {
        const filepathInfo = JSON.parse(fileItem)
        mentions.push({
          filepath: filepathInfo
        })
      } catch (e) {}
    }
  }
  return mentions
}

/**
 * Replace the placeholder with the actual file name
 * @param value
 * @returns
 */
export function replaceAtMentionPlaceHolder(value: string) {
  let newValue = value
  let match

  // Use a loop to handle cases where the string contains multiple placeholders
  while ((match = MARKDOWN_FILE_REGEX.exec(value)) !== null) {
    try {
      const filepath = match[1]
      const labelName = resolveFileNameForDisplay(filepath)
      newValue = newValue.replace(match[0], `@${labelName}`)
    } catch (error) {
      continue
    }
  }

  return newValue
}

/**
 * Encode the url in placeholder to avoid conflict with markdown syntax
 * @param value
 * @returns
 */
export function encodeMentionPlaceHolder(value: string): string {
  let newValue = value
  let match
  while ((match = PLACEHOLDER_FILE_REGEX.exec(value)) !== null) {
    try {
      newValue = newValue.replace(
        match[0],
        `[[file:${encodeURIComponent(match[1])}]]`
      )
    } catch (error) {
      continue
    }
  }
  while ((match = PLACEHOLDER_SYMBOL_REGEX.exec(value)) !== null) {
    try {
      newValue = newValue.replace(
        match[0],
        `[[symbol:${encodeURIComponent(match[1])}]]`
      )
    } catch (error) {
      continue
    }
  }

  // encode the contextCommand placeholder
  while ((match = PLACEHOLDER_COMMAND_REGEX.exec(value)) !== null) {
    try {
      newValue = newValue.replace(
        match[0],
        `[[contextCommand:${encodeURIComponent(match[1])}]]`
      )
    } catch (error) {
      continue
    }
  }

  return newValue
}

export function formatThreadTime(time: string, prefix: string) {
  const targetTime = moment(time)

  if (targetTime.isBefore(moment().subtract(1, 'year'))) {
    const timeText = targetTime.format('MMM D, YYYY')
    return `${prefix} on ${timeText}`
  }

  if (targetTime.isBefore(moment().subtract(1, 'month'))) {
    const timeText = targetTime.format('MMM D')
    return `${prefix} on ${timeText}`
  }

  return `${prefix} ${targetTime.fromNow()}`
}

export function getTitleFromMessages(
  sources: ContextSource[],
  content: string,
  options?: { maxLength?: number }
) {
  const processedContent = convertContextBlockToPlaceholder(content)
  const firstLine = processedContent.split('\n')[0] ?? ''

  const cleanedLine = firstLine
    .replace(MARKDOWN_SOURCE_REGEX, value => {
      const sourceId = value.slice(9, -2).replaceAll(/\\/g, '')
      const source = sources.find(s => s.sourceId === sourceId)
      return source?.sourceName ?? ''
    })
    .replace(PLACEHOLDER_FILE_REGEX, value => {
      try {
        const content = JSON.parse(value.slice(7, -2))
        return resolveFileNameForDisplay(content.filepath)
      } catch (e) {
        return ''
      }
    })
    .replace(PLACEHOLDER_SYMBOL_REGEX, value => {
      try {
        const content = JSON.parse(value.slice(9, -2))
        return `@${content.label}`
      } catch (e) {
        return ''
      }
    })
    .replace(PLACEHOLDER_COMMAND_REGEX, value => {
      const command = value.slice(17, -2)
      return `@${command}`
    })
    .trim()
  let title = cleanedLine
  if (options?.maxLength) {
    title = title.slice(0, options?.maxLength)
  }
  return title
}

/**
 * Process all placeholders in a message and replace them with actual content
 * @param message The original message containing placeholders
 * @param options Various handlers for different types of placeholders
 * @returns The processed message with all placeholders replaced
 */
export async function processingPlaceholder(
  message: string,
  options: {
    getChanges?: (params: GetChangesParams) => Promise<ChangeItem[]>
    readFileContent?: (info: FileRange) => Promise<string | null>
  }
): Promise<string> {
  let processedMessage = message

  // Process contextCommand placeholders
  if (options.getChanges) {
    const commandRegex = new RegExp(PLACEHOLDER_COMMAND_REGEX)
    let match
    let tempMessage = processedMessage
    while ((match = commandRegex.exec(tempMessage)) !== null) {
      const command = match[1]
      if (command === 'changes') {
        try {
          const changes = await options.getChanges({})
          const matchIndex = match.index
          const matchEnd = matchIndex + match[0].length
          const gitChanges = convertChangeItemsToContextContent(changes, {
            addPrefixNewline: shouldAddPrefixNewline(
              matchIndex,
              processedMessage
            ),
            addSuffixNewline: shouldAddSuffixNewline(matchEnd, processedMessage)
          })
          processedMessage = processedMessage.replace(match[0], gitChanges)
          tempMessage = tempMessage.replace(match[0], gitChanges)
          commandRegex.lastIndex = 0 // Reset index after replacement
        } catch (error) {
          const errorMessage = '' // Replace with empty string on error
          processedMessage = processedMessage.replace(match[0], errorMessage)
          tempMessage = tempMessage.replace(match[0], errorMessage)
          commandRegex.lastIndex = 0 // Reset index after replacement
        }
      } else {
        // Handle other commands or leave them if not supported
        // To prevent infinite loops on non-'changes' commands, ensure lastIndex advances
        // If we just continue, exec might find the same match again.
        // Simplest is to reset lastIndex, assuming replace happened or we want to skip.
        commandRegex.lastIndex = 0
      }
    }
  }

  // Process file placeholders
  if (options.readFileContent) {
    const fileRegex = new RegExp(PLACEHOLDER_FILE_REGEX)
    let match
    let tempMessage = processedMessage
    while ((match = fileRegex.exec(tempMessage)) !== null) {
      try {
        const fileInfoStr = match[1]
        const fileInfo = JSON.parse(fileInfoStr) as Filepath
        const content = await options.readFileContent({
          filepath: fileInfo,
          range: undefined
        })

        let replacement = ''
        if (content) {
          const matchIndex = match.index
          const matchEnd = matchIndex + match[0].length

          replacement = formatObjectToMarkdownBlock('file', fileInfo, content, {
            addPrefixNewline: shouldAddPrefixNewline(
              matchIndex,
              processedMessage
            ),
            addSuffixNewline: shouldAddSuffixNewline(matchEnd, processedMessage)
          })
        }

        processedMessage = processedMessage.replace(match[0], replacement)
        tempMessage = tempMessage.replace(match[0], replacement)
        fileRegex.lastIndex = 0
      } catch (error) {
        const errorMessage = `\n*Error loading file*\n`
        processedMessage = processedMessage.replace(match[0], errorMessage)
        tempMessage = tempMessage.replace(match[0], errorMessage)
        fileRegex.lastIndex = 0
      }
    }

    // Process symbol placeholders
    const symbolRegex = new RegExp(PLACEHOLDER_SYMBOL_REGEX)
    match = null // Reset match variable
    tempMessage = processedMessage // Reset tempMessage for symbol processing
    while ((match = symbolRegex.exec(tempMessage)) !== null) {
      try {
        const symbolInfoStr = match[1]
        const symbolInfo = JSON.parse(symbolInfoStr)
        const content = await options.readFileContent({
          filepath: symbolInfo.filepath,
          range: symbolInfo.range
        })

        let replacement = ''
        if (content) {
          const matchIndex = match.index
          const matchEnd = matchIndex + match[0].length

          replacement = formatObjectToMarkdownBlock(
            'symbol',
            symbolInfo,
            content,
            {
              addPrefixNewline: shouldAddPrefixNewline(
                matchIndex,
                processedMessage
              ),
              addSuffixNewline: shouldAddSuffixNewline(
                matchEnd,
                processedMessage
              )
            }
          )
        }

        processedMessage = processedMessage.replace(match[0], replacement)
        tempMessage = tempMessage.replace(match[0], replacement)
        symbolRegex.lastIndex = 0
      } catch (error) {
        const errorMessage = `\n*Error loading symbol*\n`
        processedMessage = processedMessage.replace(match[0], errorMessage)
        tempMessage = tempMessage.replace(match[0], errorMessage)
        symbolRegex.lastIndex = 0
      }
    }
  }
  return processedMessage
}

/**
 * Format markdown strings to ensure that closing tags adhere to specified newline rules
 * @param inputString
 * @returns formatted markdown string
 */
export function formatCustomHTMLBlockTags(
  inputString: string,
  tagNames: string[]
): string {
  const tagPattern = tagNames
    .map(tag => tag.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))
    .join('|')
  const regex = new RegExp(`(<(${tagPattern})>.*?</\\2>)`, 'gs')

  // Adjust the newline characters for matched closing tags
  function adjustNewlines(match: string): string {
    const startTagMatch = match.match(new RegExp(`<(${tagPattern})>`))
    const endTagMatch = match.match(new RegExp(`</(${tagPattern})>`))

    if (!startTagMatch || !endTagMatch) {
      return match
    }

    const startTag = startTagMatch[0]
    const endTag = endTagMatch[0]

    const content = match
      .slice(startTag.length, match.length - endTag.length)
      .trim()

    // One newline character before and after the start tag
    const formattedStart = `\n${startTag}\n`
    // Two newline characters before the end tag, and one after
    const formattedEnd = `\n\n${endTag}\n`

    return `${formattedStart}${content}${formattedEnd}`
  }

  return inputString.replace(regex, adjustNewlines)
}

export const normalizedMarkdownText = (input: string, maxLen?: number) => {
  const sanitizedHtml = DOMPurify.sanitize(input, {
    ALLOWED_TAGS: [],
    ALLOWED_ATTR: []
  })
  const parsed = marked.parse(sanitizedHtml) as string
  const decoded = he.decode(parsed)
  const plainText = decoded.replace(/<\/?[^>]+(>|$)/g, '')
  if (maxLen && plainText.length > maxLen) {
    return `${plainText.substring(0, maxLen)}...`
  } else {
    return plainText
  }
}

export const buildMarkdownCodeBlock = (code: string, language: string) => {
  // use ```` to avoid conflict with markdown code block
  return `\n${'````'}${language}\n${code ?? ''}\n${'````'}\n`
}

export const terminalContextToAttachmentCode = (
  context: TerminalContext
): MessageAttachmentCodeInput => {
  return {
    filepath: `terminal://${context.name}-${context.processId}`,
    content: context.selection
  }
}

export const attachmentCodeToTerminalContext = (attachmentCode: {
  filepath: string
  content: string
}): TerminalContext | undefined => {
  const { filepath, content } = attachmentCode
  if (!filepath || filepath.length === 0) {
    return undefined
  }
  let uri: URL
  try {
    uri = new URL(filepath)
  } catch (error) {
    return undefined
  }
  if (uri.protocol !== 'terminal:') {
    return undefined
  }
  const [name, processId] = uri.host.split('-')
  return {
    kind: 'terminal',
    name,
    processId: parseInt(processId),
    selection: content
  }
}
