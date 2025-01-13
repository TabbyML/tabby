// utils.ts
import { Filepath } from 'tabby-chat-panel/index'

import { FileContext } from '@/lib/types'

import { FileItem, SourceItem } from './types'

/**
 * A regular expression to match patterns like [[fileItem:{"label":"somePath","id":"123"}]].
 * This type of pattern/placeholder use the data store in database, or the data transfer between different components
 */
export const FILEITEM_REGEX = /\[\[fileItem:({.*?})\]\]/g

/**
 * A regular expression to match placeholders like [[fileItemAt: 0]] (replacements).
 * This type of pattern/placeholder use for shortening the display of the data.
 * Some markdown renderer need shorter placeholder for display.
 *
 * For example, the markdown renderer will display [[fileItemAt: 0]] as @filename 0 is unique identifier for the file. could be the index of the file in the list.
 */
export const FILEITEM_AT_REGEX = /\[\[fileItemAt: (\d+)\]\]/g

/**
 * Converts a FileItem to a SourceItem for use in the mention dropdown list.
 */
export function fileItemToSourceItem(info: FileItem): SourceItem {
  return {
    fileItem: info,
    name: getLastSegmentFromPath(info.label) || info.label, // Extract the last segment of the path as the name
    filepath: info.label,
    category: 'file'
  }
}

/**
 * Trims a label string to keep only the last suffixLength characters.
 */
export function shortenLabel(label: string, suffixLength = 15): string {
  if (label.length <= suffixLength) return label
  return '...' + label.slice(label.length - suffixLength)
}

/**
 * Replaces placeholders like [[fileItem:{"label":"xxxx"}]] with @filename for display.
 */
export function replaceAtMentionPlaceHolderWithAt(value: string) {
  let newValue = value
  let match

  // Use a loop to handle cases where the string contains multiple placeholders
  while ((match = FILEITEM_REGEX.exec(value)) !== null) {
    try {
      const parsedItem = JSON.parse(match[1])
      const labelName =
        getLastSegmentFromPath(parsedItem.label) ||
        parsedItem.label ||
        'unknown'
      newValue = newValue.replace(match[0], `@${labelName}`)
    } catch (error) {
      continue
    }
  }

  return newValue
}

interface ReplaceResult {
  newValue: string
  fileItems: FileItem[]
}

/**
 * Replaces placeholders like [[fileItem:{"label":"xxxx"}]] with [[fileItemAt: idx]]
 * and collects the corresponding FileItem objects.
 */
export function replaceAtMentionPlaceHolderWithAtPlaceHolder(
  value: string
): ReplaceResult {
  let newValue = value
  const fileItems: FileItem[] = []
  let match
  let idx = 0

  while ((match = FILEITEM_REGEX.exec(value)) !== null) {
    try {
      const parsedItem = JSON.parse(match[1])
      fileItems.push({ ...parsedItem })

      newValue = newValue.replace(match[0], `[[fileItemAt: ${idx}]]`)
      idx++
    } catch (error) {
      continue
    }
  }

  return {
    newValue,
    fileItems
  }
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
 * Retrieves the last segment from a given file path.
 */
export function getLastSegmentFromPath(filepath: string): string {
  if (!filepath) return 'unknown'
  const normalizedPath = filepath.replace(/\\/g, '/').replace(/\/+$/, '')
  const segments = normalizedPath.split('/')
  return segments[segments.length - 1] || 'unknown'
}

export function getFileBaseNameByChatPanelFilePath(filepath: Filepath): string {
  return getLastSegmentFromPath(getFilepathStringByChatPanelFilePath(filepath))
}

export function fileItemToFileContext(
  item: FileItem,
  content: string
): FileContext {
  return {
    content: content ?? '',
    filepath: getFilepathStringByChatPanelFilePath(item.filepath),
    git_url: 'filepath' in item.filepath ? item.filepath.gitUrl : '',
    kind: 'file'
  }
}
