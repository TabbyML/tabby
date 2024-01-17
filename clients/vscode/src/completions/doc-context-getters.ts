import * as vscode from 'vscode'

import { getLanguageConfig } from '../tree-sitter/language'
import { CompletionIntent, execQueryWrapper } from '../tree-sitter/query-sdk'

import { DocumentContext } from './get-current-doc-context'

export function getCurrentLinePrefixWithoutInjectedPrefix(docContext: DocumentContext): string {
    const { currentLinePrefix, injectedPrefix } = docContext

    return injectedPrefix ? currentLinePrefix.slice(0, -injectedPrefix.length) : currentLinePrefix
}

interface GetContextRangeParams {
    prefix: string
    suffix: string
    position: vscode.Position
}

/**
 * @returns the range that overlaps the included prefix and suffix.
 */
export function getContextRange(document: vscode.TextDocument, params: GetContextRangeParams): vscode.Range {
    const { prefix, suffix, position } = params
    const offset = document.offsetAt(position)

    return new vscode.Range(document.positionAt(offset - prefix.length), document.positionAt(offset + suffix.length))
}

interface GetCompletionIntentParams {
    document: vscode.TextDocument
    position: vscode.Position
    prefix: string
}

export function getCompletionIntent(params: GetCompletionIntentParams): CompletionIntent | undefined {
    const { document, position, prefix } = params

    const blockStart = getLanguageConfig(document.languageId)?.blockStart
    const isBlockStartActive = blockStart && prefix.trimEnd().endsWith(blockStart)
    // Use `blockStart` for the cursor position if it's active.
    const positionBeforeCursor = isBlockStartActive
        ? document.positionAt(prefix.lastIndexOf(blockStart))
        : {
              line: position.line,
              character: Math.max(0, position.character - 1),
          }

    const [completionIntent] = execQueryWrapper(document, positionBeforeCursor, 'getCompletionIntent')

    return completionIntent?.name
}
