import { Parent, Root, RootContent } from 'mdast'
import { remark } from 'remark'
import remarkStringify from 'remark-stringify'

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
    // TODO: extract logic for generic use in later symbol/file context change
    if (node.type === 'code' && node.meta === `label=${labelValue}`) {
      const prevNode = newChildren[newChildren.length - 1] as Parent | undefined
      const nextNode = ast.children[i + 1] as Parent | undefined

      // Determine how to insert the command based on surrounding nodes
      if (
        prevNode &&
        prevNode.type === 'paragraph' &&
        nextNode &&
        nextNode.type === 'paragraph'
      ) {
        // Case 1: Paragraphs both before and after - merge them
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
      } else if (nextNode && nextNode.type === 'paragraph') {
        // Case 2: Paragraph only after - add command before
        i++
        newChildren.push({
          type: 'paragraph',
          children: [
            { type: 'text', value: `${commandText} ` },
            ...(nextNode.children || [])
          ]
        } as RootContent)
      } else if (prevNode && prevNode.type === 'paragraph') {
        // Case 3: Paragraph only before - add command after
        ;(prevNode.children || []).push({
          type: 'text',
          value: ` ${commandText}`
        })
      } else {
        // Case 4: No paragraphs nearby - create new paragraph with command
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
 * Process context commands in text
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
  const processor = remark().use(remarkStringify as any, {
    entities: 'permissive'
  })

  const ast = processor.parse(input) as Root

  ast.children = processCodeBlocksWithLabel(ast, labelValue, commandText)
  return processor.stringify(ast)
}

export function convertContextBlockToPlaceholder(input: string): string {
  return processContextCommand(
    input,
    'changes',
    '[[contextCommand: "changes"]]'
  )
}
/**
 * convert context block to label name
 * from
 * @param input message
 * @returns
 */

export function convertContextBlockToLabelName(input: string): string {
  return processContextCommand(input, 'changes', '@changes')
}
