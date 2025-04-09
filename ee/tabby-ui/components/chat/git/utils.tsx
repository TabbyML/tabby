import { ChangeItem } from 'tabby-chat-panel/index'

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
