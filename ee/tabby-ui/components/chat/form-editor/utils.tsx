// utils.ts
import { Editor, JSONContent } from '@tiptap/core'
import { FileBox, SquareFunction } from 'lucide-react'
import { Filepath, ListSymbolItem } from 'tabby-chat-panel/index'

import {
  MARKDOWN_SOURCE_REGEX,
  PLACEHOLDER_COMMAND_REGEX,
  PLACEHOLDER_FILE_REGEX,
  PLACEHOLDER_SYMBOL_REGEX
} from '@/lib/constants/regex'
import { ContextSource } from '@/lib/gql/generates/graphql'
import { FileContext } from '@/lib/types'
import {
  convertFromFilepath,
  nanoid,
  resolveFileNameForDisplay
} from '@/lib/utils'
import { IconFile } from '@/components/ui/icons'

import { CommandItem, FileItem, SourceItem } from '../types'

/**
 * Converts a FileItem to a SourceItem for use in the mention dropdown list.
 */
export function fileItemToSourceItem(info: FileItem): SourceItem {
  const filepathString = convertFromFilepath(info.filepath).filepath
  const source: Omit<SourceItem, 'id'> = {
    fileItem: info,
    name: resolveFileNameForDisplay(filepathString), // Extract the last segment of the path as the name
    filepath: filepathString,
    category: 'file',
    icon: <IconFile />
  }
  try {
    return {
      id: JSON.stringify(info.filepath),
      ...source
    }
  } catch (e) {
    return {
      id: nanoid(),
      ...source
    }
  }
}

export function symbolItemToSourceItem(info: ListSymbolItem): SourceItem {
  const filepath = convertFromFilepath(info.filepath).filepath
  return {
    category: 'symbol',
    id: info.label,
    name: info.label,
    filepath: filepath,
    fileItem: info,
    icon: <SquareFunction className="h-4 w-4" />
  }
}

/**
 * Trims a label string to keep only the last suffixLength characters.
 */
export function shortenLabel(label: string, suffixLength = 15): string {
  if (label.length <= suffixLength) return label
  return '...' + label.slice(label.length - suffixLength)
}

export function replaceAtMentionPlaceHolderWithAt(value: string) {
  let newValue = value
  let match

  // Use a loop to handle cases where the string contains multiple placeholders
  while ((match = PLACEHOLDER_FILE_REGEX.exec(value)) !== null) {
    try {
      const filepath = match[1]
      const filepathInfo = JSON.parse(filepath) as Filepath
      const filepathString = getFilepathStringByChatPanelFilePath(filepathInfo)
      const labelName = resolveFileNameForDisplay(filepathString)
      newValue = newValue.replace(match[0], `@${labelName}`)
    } catch (error) {
      continue
    }
  }

  while ((match = PLACEHOLDER_SYMBOL_REGEX.exec(value)) !== null) {
    try {
      const symbolPlaceholder = match[1]
      const symbolInfo = JSON.parse(symbolPlaceholder)
      const labelName = symbolInfo.label || ''
      newValue = newValue.replace(match[0], `@${labelName}`)
    } catch (error) {
      continue
    }
  }

  return newValue
}

/**
 * Extracts the real file path string from a Filepath object in tabby-chat-panel.
 */
export function getFilepathStringByChatPanelFilePath(
  filepath: Filepath
): string {
  return 'filepath' in filepath ? filepath.filepath : filepath.uri
}

/**
 * convert doc mention
 * If there are dev doc mentions, convert to doc name
 */
export function convertTextToTiptapContent(
  text: string,
  sources: ContextSource[]
): JSONContent[] {
  const nodes: JSONContent[] = []
  let lastIndex = 0
  text.replace(PLACEHOLDER_FILE_REGEX, (match, filepath, offset) => {
    const content = JSON.parse(filepath) as Filepath
    const label = resolveFileNameForDisplay(
      'uri' in content ? content.uri : content.filepath
    )

    // Add text before the match as a text node
    if (offset > lastIndex) {
      nodes.push({
        type: 'text',
        text: text.slice(lastIndex, offset)
      })
    }
    try {
      // Add mention node
      nodes.push({
        type: 'mention',
        attrs: {
          category: 'file',
          fileItem: {
            filepath: JSON.parse(filepath)
          },
          label: label
        }
      })
    } catch (e) {}

    lastIndex = offset + match.length
    return match
  })

  text.replace(PLACEHOLDER_SYMBOL_REGEX, (match, symbol, offset) => {
    const content = JSON.parse(symbol) as ListSymbolItem

    const label = content.label || ''
    // Add text before the match as a text node
    if (offset > lastIndex) {
      nodes.push({
        type: 'text',
        text: text.slice(lastIndex, offset)
      })
    }
    try {
      // Add mention node
      nodes.push({
        type: 'mention',
        attrs: {
          category: 'symbol',
          fileItem: content,
          label: label
        }
      })
    } catch (e) {}
    lastIndex = offset + match.length
    return match
  })

  text.replace(PLACEHOLDER_COMMAND_REGEX, (match, command, offset) => {
    if (offset > lastIndex) {
      nodes.push({
        type: 'text',
        text: text.slice(lastIndex, offset)
      })
    }
    if (typeof command === 'string' && command.trim()) {
      nodes.push({
        type: 'mention',
        attrs: {
          category: 'command',
          command: command.trim(),
          label: command.trim()
        }
      })
    }

    lastIndex = offset + match.length
    return match
  })

  text.replace(MARKDOWN_SOURCE_REGEX, (match, sourceId, offset) => {
    if (offset > lastIndex) {
      nodes.push({
        type: 'text',
        text: text.slice(lastIndex, offset)
      })
    }
    const source = sourceId
      ? sources.find(x => x.sourceId === sourceId)
      : undefined
    if (source) {
      nodes.push({
        type: 'text',
        text: source.sourceName
      })
    }

    lastIndex = offset + match.length
    return match
  })

  // Add remaining text as a text node
  if (lastIndex < text.length) {
    nodes.push({
      type: 'text',
      text: text.slice(lastIndex)
    })
  }

  return nodes
}

/**
 * Checks if two file contexts refer to the same entire file.
 * In this scenario, contexts without a range refer to the entire file.
 *
 * @param fileContext1 - The first file context to compare.
 * @param fileContext2 - The second file context to compare.
 * @returns {boolean} - Returns true if both file contexts refer to the same entire file; otherwise, false.
 */
export function isSameEntireFileContextFromMention(
  fileContext1: FileContext,
  fileContext2: FileContext
) {
  return (
    fileContext1.filepath === fileContext2.filepath &&
    fileContext1.baseDir === fileContext2.baseDir &&
    fileContext1.gitUrl === fileContext2.gitUrl &&
    !fileContext1.range &&
    !fileContext2.range
  )
}

export const isSameFileContext = (a: FileContext, b: FileContext) => {
  const sameBasicInfo =
    a.filepath === b.filepath &&
    a.baseDir === b.baseDir &&
    a.gitUrl === b.gitUrl

  if (!a.range && !b.range) return sameBasicInfo

  return (
    sameBasicInfo &&
    a.range?.start === b.range?.start &&
    a.range?.end === b.range?.end
  )
}

export function getMention(editor: Editor) {
  const currentMentions: any[] = []
  editor.state.doc.descendants(node => {
    if (node.type.name === 'mention') {
      currentMentions.push(node.attrs)
    }
  })
  return currentMentions
}

export function commandItemToSourceItem(info: CommandItem): SourceItem {
  return {
    id: info.id,
    name: info.name,
    category: 'command',
    command: info.command,
    description: info.description,
    icon: <FileBox className="h-4 w-4" />
  }
}

/**
 * Creates a default "changes" command item.
 */
export function createChangesCommand(): CommandItem {
  return {
    id: 'changes',
    name: 'changes',
    command: 'changes',
    description: 'Adding git diff changes into context'
  }
}
