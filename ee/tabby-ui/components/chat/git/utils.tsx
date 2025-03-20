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
  const startIndex = text.indexOf('[[contextCommand:"')
  if (startIndex === -1) return null
  const commandStartIndex = startIndex + '[[contextCommand:"'.length
  const commandEndIndex = text.indexOf('"]]', commandStartIndex)
  if (commandEndIndex === -1) return null
  return text.slice(commandStartIndex, commandEndIndex)
}

/**
 * Converts a GitChange object to a MessageAttachmentCodeInput format suitable for API requests.
 * The function wraps the diff content in a code block with a specific label to identify it as a git change.
 *
 * @param gitChange - The GitChange object containing diff information
 * @returns A MessageAttachmentCodeInput object with formatted content
 *
 * @example
 * // Input GitChange
 * const gitChange = {
 *   id: "abc123",
 *   filepath: "src/components/Button.tsx",
 *   additions: 15,
 *   deletions: 7,
 *   diffContent: "diff --git a/src/components/Button.tsx b/src/components/Button.tsx\nindex 1234567..abcdef0 100644\n--- a/src/components/Button.tsx\n+++ b/src/components/Button.tsx\n@@ -10,7 +10,15 @@ export const Button = () => {\n   return (\n-    <button className=\"btn\">\n+    <button className=\"btn btn-primary\">\n       Click me\n     </button>\n   );\n }",
 *   lineStart: 10
 * };
 * const output = {
 *   content: "```diff label=changes\ndiff --git a/src/components/Button.tsx b/src/components/Button.tsx\nindex 1234567..abcdef0 100644\n--- a/src/components/Button.tsx\n+++ b/src/components/Button.tsx\n@@ -10,7 +10,15 @@ export const Button = () => {\n   return (\n-    <button className=\"btn\">\n+    <button className=\"btn btn-primary\">\n       Click me\n     </button>\n   );\n } ```",
 *   filepath: "src/components/Button.tsx",
 *   startLine: 10
 * }
 */
export function convertGitChangesToContextContent(
  gitChange: GitChange
): string {
  const { diffContent, filepath } = gitChange
  return `\`\`context command=git_diff_changes filepath:${filepath}\n${diffContent}\n\`\``
}
