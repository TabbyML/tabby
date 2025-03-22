import { uniq } from 'lodash-es'
import moment from 'moment'
import type {
  ChangeItem,
  Filepath,
  FileRange,
  GetChangesParams
} from 'tabby-chat-panel'

import {
  ContextInfo,
  ContextSource,
  ContextSourceKind
} from '@/lib/gql/generates/graphql'
import type { MentionAttributes } from '@/lib/types'
import {
  convertChangeItemsToContextContent,
  hasChangesCommand
} from '@/components/chat/git/utils'

import {
  MARKDOWN_FILE_REGEX,
  MARKDOWN_SOURCE_REGEX,
  PLACEHOLDER_COMMAND_REGEX,
  PLACEHOLDER_FILE_REGEX,
  PLACEHOLDER_SYMBOL_REGEX
} from '../constants/regex'
import { convertContextBlockToPlaceholder } from './markdown'

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
        `[[contextCommand:"${encodeURIComponent(match[1])}"]]`
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
      const command = value.slice(18, -3)
      return `@${command}`
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
      const command = value.slice(19, -3)
      return `@${command}`
    })
    .trim()

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
  if (hasChangesCommand(processedMessage) && options.getChanges) {
    try {
      const changes = await options.getChanges({})
      const gitChanges = convertChangeItemsToContextContent(changes)
      processedMessage = processedMessage.replaceAll(
        /\[\[contextCommand:"changes"\]\]/g,
        gitChanges
      )
    } catch (error) {
      processedMessage = processedMessage.replaceAll(
        /\[\[contextCommand:"changes"\]\]/g,
        ''
      )
    }
  }
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
          const fileInfoJSON = JSON.stringify(fileInfo).replace(/"/g, '\\"')
          replacement = `\n\`\`\`context label='file' object='${fileInfoJSON}'\n${content}\n\`\`\`\n`
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

    const symbolRegex = new RegExp(PLACEHOLDER_SYMBOL_REGEX)
    match = null
    tempMessage = processedMessage
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
          const symbolInfoJSON = JSON.stringify(symbolInfo).replace(/"/g, '\\"')
          replacement = `\n\`\`\`context label='symbol' object='${symbolInfoJSON}'\n${content}\n\`\`\`\n`
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
