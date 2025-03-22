import { Parent, Root, RootContent } from 'mdast'
import { remark } from 'remark'
import remarkStringify from 'remark-stringify'

/**
 * Custom stringification of AST to preserve special patterns
 * @param ast AST to stringify
 * @returns Plain string representation
 */
export function customAstToString(ast: Root): string {
  let result = ''

  for (const node of ast.children) {
    result += nodeToString(node) + '\n'
  }

  // Remove trailing newlines
  return result.trim()
}

/**
 * Convert a single node to string
 * @param node AST node
 * @returns String representation
 */
function nodeToString(node: any): string {
  switch (node.type) {
    case 'paragraph':
      return paragraphToString(node)
    case 'text':
      return node.value
    default:
      const processor = remark().use(remarkStringify)
      return processor.stringify({ type: 'root', children: [node] }).trim()
  }
}

/**
 * Convert paragraph node to string
 * @param node Paragraph node
 * @returns String representation
 */
function paragraphToString(node: any): string {
  return childrenToString(node)
}

/**
 * Process children of a node and join them
 * @param node Parent node
 * @returns Combined string of all children
 */
function childrenToString(node: any): string {
  if (!node.children || node.children.length === 0) {
    return ''
  }

  return node.children.map((child: any) => nodeToString(child)).join('')
}

/**
 * Process code blocks with specific labels and insert corresponding command markers
 * @param ast The parsed AST
 * @param labelValue Label value to look for
 * @param commandText Command text to insert
 * @returns Processed AST children array
 */
export function processCodeBlocksWithLabel(
  ast: Root,
  labelValue: string,
  commandText: string
): RootContent[] {
  const newChildren: RootContent[] = []
  for (let i = 0; i < ast.children.length; i++) {
    const node = ast.children[i]
    // Check if the node is a code block with the specified label
    if (node.type === 'code' && node.meta === `label=${labelValue}`) {
      const prevNode = newChildren[newChildren.length - 1] as Parent | undefined
      const nextNode = ast.children[i + 1] as Parent | undefined

      // Check if nodes are on the same line using position data
      const isPrevNodeSameLine =
        prevNode &&
        prevNode.position &&
        node.position &&
        node.position.start.line - prevNode.position.end.line === 1

      const isNextNodeSameLine =
        nextNode &&
        nextNode.position &&
        node.position &&
        nextNode.position.start.line - node.position.end.line === 1

      // Determine how to insert the command based on surrounding nodes and line positions
      if (
        prevNode &&
        prevNode.type === 'paragraph' &&
        nextNode &&
        nextNode.type === 'paragraph' &&
        isPrevNodeSameLine &&
        isNextNodeSameLine
      ) {
        // Case 1: Paragraphs both before and after on same lines - merge them
        i++
        newChildren.pop()
        newChildren.push({
          type: 'paragraph',
          children: [
            ...(prevNode.children || []),
            { type: 'text', value: ` ${commandText} ` },
            ...(nextNode.children || [])
          ]
        } as RootContent)
      } else if (
        nextNode &&
        nextNode.type === 'paragraph' &&
        isNextNodeSameLine
      ) {
        // Case 2: Paragraph only after on same line - add command before
        i++
        newChildren.push({
          type: 'paragraph',
          children: [
            { type: 'text', value: `${commandText} ` },
            ...(nextNode.children || [])
          ]
        } as RootContent)
      } else if (
        prevNode &&
        prevNode.type === 'paragraph' &&
        isPrevNodeSameLine
      ) {
        // Case 3: Paragraph only before on same line - add command after
        ;(prevNode.children || []).push({
          type: 'text',
          value: ` ${commandText}`
        })
      } else {
        // Case 4: No paragraphs nearby on same lines - create new paragraph with command
        newChildren.push({
          type: 'paragraph',
          children: [{ type: 'text', value: commandText }]
        } as RootContent)
      }
    } else {
      // Non-matching nodes remain unchanged
      newChildren.push(node)
    }
  }
  return newChildren
}

/**
 * Process context commands in text using custom AST string conversion
 * @param input Input text
 * @param labelValue Label value to look for
 * @param commandText Command text to insert
 * @returns Processed text
 */
export function processContextCommand(
  input: string,
  labelValue: string,
  commandText: string
): string {
  // Parse input to AST
  const processor = remark()
  const ast = processor.parse(input) as Root

  // Process the AST using our AST-based approach
  ast.children = processCodeBlocksWithLabel(ast, labelValue, commandText)

  // Use our custom stringifier instead of the built-in one
  return customAstToString(ast)
}

/**
 * Convert context block to placeholder
 * @param input Input text
 * @returns Processed text with context commands replaced by placeholders
 */
export function convertContextBlockToPlaceholder(input: string): string {
  return processContextCommand(
    input,
    'changes',
    '[[contextCommand: "changes"]]'
  )
}

/**
 * Convert context block to label name
 * @param input Input text
 * @returns Processed text with context commands replaced by label names
 */
export function convertContextBlockToLabelName(input: string): string {
  return processContextCommand(input, 'changes', '@changes')
}
