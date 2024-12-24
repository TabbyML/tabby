import { AtInfo, FileAtInfo } from 'tabby-chat-panel/index'

import { MentionNodeAttrs, SourceItem } from './types'

/**
 * A regex to detect patterns like [[atSource: {...}]]
 * The JSON inside can be parsed to reconstruct AtInfo data.
 */
export const AT_SOURCE_REGEX = /\[\[atSource:(.*?)\]\]/g

/**
 * Type guard to check if the given AtInfo is a FileAtInfo.
 */
export function isFileAtInfo(atInfo: AtInfo): atInfo is FileAtInfo {
  return atInfo.atKind === 'file'
}

/**
 * Convert an AtInfo object into a SourceItem for display and mention.
 * @param info An AtInfo object from tabby-chat-panel
 * @returns A SourceItem containing category, name, filepath, and the raw atInfo
 */
export function atInfoToSourceItem(info: AtInfo): SourceItem {
  if (isFileAtInfo(info)) {
    return {
      category: 'files' as const,
      atInfo: info,
      name: info.name,
      filepath:
        'uri' in info.filepath ? info.filepath.uri : info.filepath.filepath
    }
  } else {
    return {
      category: 'symbols' as const,
      atInfo: info,
      name: info.name,
      filepath:
        'uri' in info.location.filepath
          ? info.location.filepath.uri
          : info.location.filepath.filepath
    }
  }
}

/**
 * Convert a SourceItem into a MentionNodeAttrs object to be used by Tiptap.
 * Useful for inserting a mention node into the editor.
 * @param item The SourceItem to convert
 * @returns The mention node attributes needed by Tiptap
 */
export function sourceItemToMentionNodeAttrs(
  item: SourceItem
): MentionNodeAttrs {
  return {
    id: `${item.name}-${item.filepath}`,
    name: item.name,
    category: item.category,
    atInfo: item.atInfo!
  }
}

/**
 * Extracts AtInfo objects from text that match the AT_SOURCE_REGEX pattern.
 * Replaces the matched patterns with @<parsedData.name> in the original text.
 * @param text The text to parse
 * @returns An object with updated text (after replacement) and a list of extracted AtInfo
 */
export function extractAtSourceFromString(text: string) {
  const atInfos: AtInfo[] = []
  let match

  while ((match = AT_SOURCE_REGEX.exec(text))) {
    const sourceData = match[1]
    try {
      const parsedAtInfo = JSON.parse(sourceData)
      atInfos.push(parsedAtInfo)
      text = text.replace(match[0], `@${parsedAtInfo.name}`)
    } catch {
      // If JSON parsing fails, skip this match
      continue
    }
  }

  return { text, atInfos }
}
