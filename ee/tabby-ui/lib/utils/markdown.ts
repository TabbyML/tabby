import { Parent, Root, RootContent } from 'mdast'
import { remark } from 'remark'
import remarkStringify, { Options } from 'remark-stringify'

const REMARK_STRINGIFY_OPTIONS: Options = {
  bullet: '*',
  emphasis: '*',
  fences: true,
  listItemIndent: 'one',
  tightDefinitions: true,
  handlers: {
    placeholder: (node: PlaceholderNode) => {
      return node.value
    }
  } as any
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
  const processor = createRemarkProcessor()
  return processor.stringify(ast).trim()
}

/**
 * Process code blocks with labels and convert them to placeholders
 */
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
      let placeholderNode: RootContent | null = null
      let shouldProcessNode = true

      switch (metas['label']) {
        case 'changes':
          finalCommandText = '"changes"'
          placeholderNode = createPlaceholderNode(
            `[[contextCommand:${finalCommandText}]]`
          ) as unknown as RootContent
          break
        case 'file':
          if (metas['object']) {
            try {
              placeholderNode = createPlaceholderNode(
                `[[file:${metas['object']}]]`
              ) as unknown as RootContent
              if (!placeholderNode) {
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
              placeholderNode = createPlaceholderNode(
                `[[symbol:${metas['object']}]]`
              ) as unknown as RootContent
              if (!placeholderNode) {
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
            placeholderNode || { type: 'text', value: ` ${finalCommandText} ` },
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
            placeholderNode || { type: 'text', value: `${finalCommandText} ` },
            ...(nextNode.children || [])
          ]
        } as RootContent)
      } else if (
        prevNode &&
        prevNode.type === 'paragraph' &&
        isPrevNodeSameLine
      ) {
        ;(prevNode.children || []).push(
          placeholderNode || { type: 'text', value: ` ${finalCommandText}` }
        )
      } else {
        newChildren.push({
          type: 'paragraph',
          children: [
            placeholderNode || { type: 'text', value: finalCommandText }
          ]
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
    return res
  } catch (error) {
    return `\n*Error formatting ${label}*\n`
  }
}

export interface PlaceholderNode extends Node {
  type: 'placeholder'
  value: string
}

export function createPlaceholderNode(value: string): PlaceholderNode {
  return {
    type: 'placeholder',
    value: value
  } as PlaceholderNode
}
