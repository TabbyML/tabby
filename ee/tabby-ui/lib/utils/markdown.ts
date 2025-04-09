import { Root, RootContent, Text } from 'mdast'
import { remark } from 'remark'
import remarkStringify, { Options } from 'remark-stringify'
import { ChangeItem } from 'tabby-chat-panel/index'

import { remarkCodeBlocksToPlaceholders } from './markdown/remark-codeblock-to-placeholder'
import {
  PlaceholderNode,
  placeholderToString,
  remarkPlaceholderParser
} from './markdown/remark-placeholder-parser'

const REMARK_STRINGIFY_OPTIONS: Options = {
  bullet: '*',
  emphasis: '*',
  fences: true,
  listItemIndent: 'one',
  tightDefinitions: true,
  handlers: {
    placeholder: (node: PlaceholderNode) => {
      // It's should create a formatted plugin for this, but for now, it's just a simple function
      return placeholderToString(node)
    },
    text: (node: Text) => {
      return node.value
    }
  } as any
}

function createRemarkProcessor() {
  return remark()
    .use(remarkPlaceholderParser)
    .use(remarkCodeBlocksToPlaceholders)
    .use(remarkStringify, REMARK_STRINGIFY_OPTIONS)
}

/**
 * Custom stringification of AST using remarkStringify
 * @param ast AST to stringify
 * @returns Plain string representation
 */
export function customAstToString(ast: Root): string {
  const processor = createRemarkProcessor()
  return processor.stringify(ast).trim()
}

/**
 * Process markdown text with context commands and convert them to placeholders
 * @param input The markdown text to process
 * @returns Processed markdown with placeholders
 */
export function parseMarkdownWithContextCommands(input: string): string {
  const processor = createRemarkProcessor()
  const ast = processor.runSync(processor.parse(input)) as Root
  return customAstToString(ast)
}

/**
 * Convert context blocks in markdown to placeholder nodes (legacy function name)
 * @param input The markdown text containing context blocks
 * @returns Processed markdown with placeholders
 */
export function convertContextBlockToPlaceholder(input: string): string {
  return parseMarkdownWithContextCommands(input)
}

/**
 * Format an object into a markdown code block with proper metadata
 * @param label The label for the code block (e.g., 'file', 'symbol')
 * @param obj The object to format
 * @param content The content to include in the code block
 * @param options Optional configuration for formatting
 * @returns A formatted markdown code block string
 */
export function formatObjectToMarkdownBlock(
  label: string,
  obj: any,
  content: string,
  options?: {
    addPrefixNewline?: boolean
    addSuffixNewline?: boolean
  }
): string {
  try {
    const { addPrefixNewline = true, addSuffixNewline = true } = options || {}
    const metaObj = {
      label: label,
      object: obj
    }
    const metaJSON = JSON.stringify(metaObj)

    const codeNode: Root = {
      type: 'root',
      children: [
        {
          type: 'code',
          lang: 'context',
          meta: metaJSON,
          value: content
        } as RootContent
      ]
    }

    const processor = createRemarkProcessor()
    const formattedContent = processor.stringify(codeNode).trim()

    const prefix = addPrefixNewline ? '\n' : ''
    const suffix = addSuffixNewline ? '\n' : ''

    return `${prefix}${formattedContent}${suffix}`
  } catch (error) {
    const { addPrefixNewline = true, addSuffixNewline = true } = options || {}
    return `${addPrefixNewline ? '\n' : ''}*Error formatting ${label}*${
      addSuffixNewline ? '\n' : ''
    }`
  }
}

export function convertChangeItemsToContextContent(
  changes: ChangeItem[],
  options?: { addPrefixNewline?: boolean; addSuffixNewline?: boolean }
): string {
  const content = changes.map(change => change.content).join('\\n')
  const { addPrefixNewline = true, addSuffixNewline = true } = options || {}

  const meta = {
    label: 'changes'
  }
  const codeNode: Root = {
    type: 'root',
    children: [
      {
        type: 'code',
        lang: 'context',
        meta: JSON.stringify(meta),
        value: content
      } as RootContent
    ]
  }

  const processor = createRemarkProcessor()
  let formattedContent = processor.stringify(codeNode).trim()

  const prefix = addPrefixNewline ? '\n' : ''
  const suffix = addSuffixNewline ? '\n' : ''

  return `${prefix}${formattedContent}${suffix}`
}

/**
 * Determines if a prefix newline should be added based on the context
 * @param index The starting index of the placeholder in the text
 * @param text The full text containing the placeholder
 * @returns Whether a prefix newline should be added
 */
export function shouldAddPrefixNewline(index: number, text: string): boolean {
  if (index === 0) return false

  let i = index - 1
  while (i >= 0) {
    if (text[i] === '\n') return false
    if (text[i] === '\r' && i + 1 < text.length && text[i + 1] === '\n') {
      return false
    }

    if (!/\s/.test(text[i])) return true

    i--
  }

  return false
}

/**
 * Determines if a suffix newline should be added based on the context
 * @param index The ending index of the placeholder in the text
 * @param text The full text containing the placeholder
 * @returns Whether a suffix newline should be added
 */
export function shouldAddSuffixNewline(index: number, text: string): boolean {
  const len = text.length
  if (index >= len) return false

  let i = index
  while (i < len) {
    if (text[i] === '\n') return false
    if (text[i] === '\r' && i + 1 < len && text[i + 1] === '\n') return false

    if (!/\s/.test(text[i])) return true
    i++
  }

  return false
}
