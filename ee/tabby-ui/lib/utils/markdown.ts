import { Parent, Root, RootContent } from 'mdast'
import { remark } from 'remark'
import remarkStringify from 'remark-stringify'
import { Options } from 'remark-stringify'

const REMARK_STRINGIFY_OPTIONS: Options = {
  bullet: '*',
  emphasis: '*',
  fences: true,
  listItemIndent: 'one',
  tightDefinitions: true
}


function createRemarkProcessor() {
  return remark().use(remarkStringify, REMARK_STRINGIFY_OPTIONS)
}

/**
 * Custom stringification of AST using remarkStringify
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
      return createRemarkProcessor().stringify({ type: 'root', children: [node] }).trim()
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
        metas[key] = rawValue
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
      let shouldProcessNode = true

      // processing differet type of context
      switch (metas['label']) {
        case 'changes':
          finalCommandText = '[[contextCommand:"changes"]]'
          break
        case 'file':
          if (metas['object']) {
            try {
              finalCommandText = formatPlaceholder('file', metas['object'])
              if (!finalCommandText) {
                shouldProcessNode = false
                newChildren.push(node)
              }
            } catch (error) {
              shouldProcessNode = false
              newChildren.push(node)
            }
          }
          break
        case 'symbol':
          if (metas['object']) {
            try {
              finalCommandText = formatPlaceholder('symbol', metas['object'])
              if (!finalCommandText) {
                shouldProcessNode = false
                newChildren.push(node)
              }
            } catch (error) {
              shouldProcessNode = false
              newChildren.push(node)
            }
          }
          break
        default:
          newChildren.push(node)
          continue
      }

      if (!shouldProcessNode) {
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
  const processor = createRemarkProcessor()
  const ast = processor.parse(input) as Root
  ast.children = processCodeBlocksWithLabel(ast)
  return customAstToString(ast)
}

export function convertContextBlockToPlaceholder(input: string): string {
  return processContextCommand(input)
}

/**
 * Format an object into a markdown code block with proper metadata
 * @param label The label for the code block (e.g., 'file', 'symbol')
 * @param obj The object to format
 * @param content The content to include in the code block
 * @returns A formatted markdown code block string
 */
export function formatObjectToMarkdownBlock(
  label: string,
  obj: any,
  content: string
): string {
  try {
    // Convert the object to a JSON string
    const objJSON = JSON.stringify(obj)

    const codeNode: Root = {
      type: 'root',
      children: [
        {
          type: 'code',
          lang: 'context',
          meta: `label=${label} object=${objJSON}`,
          value: content
        } as RootContent
      ]
    }
    
    const processor = createRemarkProcessor()
    
    const res = '\n' + processor.stringify(codeNode).trim() + '\n'
    console.log('res', res)
    return res;
  } catch (error) {
    console.error(`Error formatting ${label} to markdown block:`, error)
    return `\n*Error formatting ${label}*\n`
  }
}



/**
 * Format a placeholder with proper backslash escaping
 * @param type The type of placeholder (e.g., 'file', 'symbol')
 * @param objStr The string representation of the object
 * @returns The formatted placeholder text
 */
export function formatPlaceholder(type: string, objStr: string): string {
  if (!objStr) return ''
  
  try {
    return `[[${type}:${objStr}]]`
  } catch (error) {
    console.error(`Error formatting ${type} placeholder:`, error)
    return ''
  }
}
