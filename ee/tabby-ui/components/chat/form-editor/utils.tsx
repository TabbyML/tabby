// utils.ts
import { JSONContent } from '@tiptap/core'
import { SquareFunction } from 'lucide-react'
import { Filepath, ListSymbolItem } from 'tabby-chat-panel/index'

import { PLACEHOLDER_FILE_REGEX } from '@/lib/constants/regex'
import { FileContext } from '@/lib/types'
import { convertFilepath, nanoid, resolveFileNameForDisplay } from '@/lib/utils'
import { IconFile } from '@/components/ui/icons'

import { FileItem, SourceItem } from './types'

/**
 * Converts a FileItem to a SourceItem for use in the mention dropdown list.
 */
export function fileItemToSourceItem(info: FileItem): SourceItem {
  const filepathString = convertFilepath(info.filepath).filepath
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
  const filepath = convertFilepath(info.filepath).filepath
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

export function convertTextToTiptapContent(text: string): JSONContent[] {
  const nodes: JSONContent[] = []
  let lastIndex = 0
  text.replace(PLACEHOLDER_FILE_REGEX, (match, filepath, offset) => {
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
          }
        }
      })
    } catch (e) {}

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
    fileContext1.git_url === fileContext2.git_url &&
    !fileContext1.range &&
    !fileContext2.range
  )
}

export const isSameFileContext = (a: FileContext, b: FileContext) => {
  const sameBasicInfo = a.filepath === b.filepath && a.git_url === b.git_url

  if (!a.range && !b.range) return sameBasicInfo

  return (
    sameBasicInfo &&
    a.range?.start === b.range?.start &&
    a.range?.end === b.range?.end
  )
}
