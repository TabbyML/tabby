import { ChangeItem } from 'tabby-chat-panel/index'

import { PLACEHOLDER_COMMAND_REGEX } from '@/lib/constants/regex'
import { nanoid } from '@/lib/utils'

import { GitChange } from '../types'

/**
 * Parse a git diff string and extract file changes
 */
export function convertChangesToGitChanges(changes: ChangeItem[]): GitChange[] {
  const gitChanges: GitChange[] = []

  for (const change of changes) {
    const diffLines = change.content.split('\n')

    const filePathLine = diffLines.find(line => line.startsWith('diff --git'))
    if (!filePathLine) continue

    const filePathMatch = filePathLine.match(/diff --git a\/(.*?) b\/(.*)/)
    if (!filePathMatch || !filePathMatch[1]) continue

    const filepath = filePathMatch[1]

    const statLine = diffLines.find(line => line.startsWith('@@'))
    if (!statLine) continue

    const statMatch = statLine.match(/@@ -(\d+),?(\d*) \+(\d+),?(\d*) @@/)
    if (!statMatch) continue

    const deletions = statMatch[2] ? parseInt(statMatch[2], 10) : 0
    const additions = statMatch[4] ? parseInt(statMatch[4], 10) : 0

    // Extract the starting line number
    const lineStart = parseInt(statMatch[3], 10)

    gitChanges.push({
      id: nanoid(),
      filepath,
      additions,
      deletions,
      diffContent: change.content,
      lineStart
    })
  }

  return gitChanges
}

export function hasChangesCommand(text: string): boolean {
  const commandMatch = text.match(PLACEHOLDER_COMMAND_REGEX)
  return (
    commandMatch !== null &&
    extractContextCommand(commandMatch[0]) === 'changes'
  )
}

export function extractContextCommand(text: string): string | null {
  const startIndex = text.indexOf('[[contextCommand:')
  if (startIndex === -1) return null
  const commandStartIndex = startIndex + '[[contextCommand:'.length
  const commandEndIndex = text.indexOf(']]', commandStartIndex)
  if (commandEndIndex === -1) return null
  return text.slice(commandStartIndex, commandEndIndex)
}

export function convertChangeItemsToContextContent(
  changes: ChangeItem[]
): string {
  const prefix = '\n```diff label=changes\n'
  const suffix = '\n```'
  const content = changes.map(change => change.content).join('\n')
  return `${prefix}${content}${suffix}\n`
}
