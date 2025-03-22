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

export function processCodeBlocksWithLabel(ast: Root): RootContent[] {
  const newChildren: RootContent[] = []
  for (let i = 0; i < ast.children.length; i++) {
    const node = ast.children[i]
    const metas: Record<string, string> = {}

    if (node.type === 'code' && node.meta) {
      node.meta?.split(' ').forEach(item => {
        const [key, rawValue] = item.split(/=(.+)/)
        const value = rawValue?.replace(/^['"]|['"]$/g, '') || ''
        metas[key] = value
      })
    }

    if (node.type === 'code' && metas['label']) {
      const prevNode = newChildren[newChildren.length - 1] as Parent | undefined
      const nextNode = ast.children[i + 1] as Parent | undefined

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

      let finalCommandText = ''

      // processing differet type of context
      switch (metas['label']) {
        case 'changes':
          finalCommandText = '[[contextCommand:"changes"]]'
          break
        case 'file':
          if (metas['object']) {
            const fileObject = JSON.parse(metas['object'].replace(/\\"/g, '"'))
            finalCommandText = `[[file:${JSON.stringify(fileObject)}]]`
          }
          break
        case 'symbol':
          if (metas['object']) {
            const symbolObject = JSON.parse(
              metas['object'].replace(/\\"/g, '"')
            )
            finalCommandText = `[[symbol:${JSON.stringify(symbolObject)}]]`
          }
          break
        default:
          newChildren.push(node)
          continue
      }

      if (
        prevNode &&
        prevNode.type === 'paragraph' &&
        nextNode &&
        nextNode.type === 'paragraph' &&
        isPrevNodeSameLine &&
        isNextNodeSameLine
      ) {
        i++
        newChildren.pop()
        newChildren.push({
          type: 'paragraph',
          children: [
            ...(prevNode.children || []),
            { type: 'text', value: ` ${finalCommandText} ` },
            ...(nextNode.children || [])
          ]
        } as RootContent)
      } else if (
        nextNode &&
        nextNode.type === 'paragraph' &&
        isNextNodeSameLine
      ) {
        i++
        newChildren.push({
          type: 'paragraph',
          children: [
            { type: 'text', value: `${finalCommandText} ` },
            ...(nextNode.children || [])
          ]
        } as RootContent)
      } else if (
        prevNode &&
        prevNode.type === 'paragraph' &&
        isPrevNodeSameLine
      ) {
        ;(prevNode.children || []).push({
          type: 'text',
          value: ` ${finalCommandText}`
        })
      } else {
        newChildren.push({
          type: 'paragraph',
          children: [{ type: 'text', value: finalCommandText }]
        } as RootContent)
      }
    } else {
      newChildren.push(node)
    }
  }
  return newChildren
}

export function processContextCommand(input: string): string {
  const processor = remark()
  const ast = processor.parse(input) as Root
  ast.children = processCodeBlocksWithLabel(ast)
  return customAstToString(ast)
}

export function convertContextBlockToPlaceholder(input: string): string {
  return processContextCommand(input)
}
