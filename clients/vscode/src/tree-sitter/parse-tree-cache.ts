import { LRUCache } from 'lru-cache'
import * as vscode from 'vscode'
import { TextDocument } from 'vscode'
import Parser, { Tree } from 'web-tree-sitter'

import { getParseLanguage, SupportedLanguage } from './grammars'
import { createParser, getParser } from './parser'

const parseTreesPerFile = new LRUCache<string, Tree>({
    max: 10,
})

interface ParseTreeCache {
    tree: Tree
    parser: Parser
    cacheKey: string
}

export function getCachedParseTreeForDocument(document: TextDocument): ParseTreeCache | null {
    const parseLanguage = getLanguageIfTreeSitterEnabled(document)

    if (!parseLanguage) {
        return null
    }

    const parser = getParser(parseLanguage)
    const cacheKey = document.uri.toString()
    const tree = parseTreesPerFile.get(cacheKey)

    if (!tree || !parser) {
        return null
    }

    return { tree, parser, cacheKey }
}

export async function parseDocument(document: TextDocument): Promise<void> {
    const parseLanguage = getLanguageIfTreeSitterEnabled(document)

    if (!parseLanguage) {
        return
    }

    const parser = await createParser({ language: parseLanguage })
    updateParseTreeCache(document, parser)
}

export function updateParseTreeCache(document: TextDocument, parser: Parser): void {
    const tree = parser.parse(document.getText())
    parseTreesPerFile.set(document.uri.toString(), tree)
}

function getLanguageIfTreeSitterEnabled(document: TextDocument): SupportedLanguage | null {
    const parseLanguage = getParseLanguage(document.languageId)

    /**
     * 1. Do not use tree-sitter for unsupported languages.
     * 2. Do not use tree-sitter for files with more than N lines to avoid performance issues.
     *    - https://github.com/tree-sitter/tree-sitter/issues/2144
     *    - https://github.com/neovim/neovim/issues/22426
     *
     *    Needs more testing to figure out if we need it. Playing it safe for the initial integration.
     */
    if (document.lineCount <= 10_000 && parseLanguage) {
        return parseLanguage
    }

    return null
}

export function updateParseTreeOnEdit(edit: vscode.TextDocumentChangeEvent): void {
    const { document, contentChanges } = edit
    if (contentChanges.length === 0) {
        return
    }

    const cache = getCachedParseTreeForDocument(document)
    if (!cache) {
        return
    }

    const { tree, parser, cacheKey } = cache

    for (const change of contentChanges) {
        const startIndex = change.rangeOffset
        const oldEndIndex = change.rangeOffset + change.rangeLength
        const newEndIndex = change.rangeOffset + change.text.length
        const startPosition = document.positionAt(startIndex)
        const oldEndPosition = document.positionAt(oldEndIndex)
        const newEndPosition = document.positionAt(newEndIndex)
        const startPoint = asPoint(startPosition)
        const oldEndPoint = asPoint(oldEndPosition)
        const newEndPoint = asPoint(newEndPosition)

        tree.edit({
            startIndex,
            oldEndIndex,
            newEndIndex,
            startPosition: startPoint,
            oldEndPosition: oldEndPoint,
            newEndPosition: newEndPoint,
        })
    }

    const updatedTree = parser.parse(document.getText(), tree)
    parseTreesPerFile.set(cacheKey, updatedTree)
}

export function asPoint(position: Pick<vscode.Position, 'line' | 'character'>): Parser.Point {
    return { row: position.line, column: position.character }
}

export function parseAllVisibleDocuments(): void {
    for (const editor of vscode.window.visibleTextEditors) {
        void parseDocument(editor.document)
    }
}
