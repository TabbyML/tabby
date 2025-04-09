/**
 * Transformer plugin for processing code blocks with labels and converting them to placeholders
 */

import { Code, Parent, Root, RootContent } from 'mdast'
import { Plugin } from 'unified'

import { PlaceholderNode } from './remark-placeholder-parser'

/**
 * Creates a placeholder node from a given type and object
 */
export function createPlaceholderNode(
  placeholderType: string,
  obj: string
): PlaceholderNode {
  return {
    type: 'placeholder',
    placeholderType: placeholderType,
    attributes: {
      object: obj
    }
  } as PlaceholderNode
}

/**
 * Process metadata from code blocks and extract label and object information
 */
export function parseCodeBlockMeta(
  meta: string | null | undefined
): Record<string, any> {
  if (!meta) {
    return {}
  }

  try {
    const parsedMeta = JSON.parse(meta)
    // Ensure the parsed result is an object, although JSON.parse usually returns one
    if (typeof parsedMeta === 'object' && parsedMeta !== null) {
      return parsedMeta
    }
    return {}
  } catch (error) {
    return {}
  }
}

/**
 * Remark plugin for processing code blocks with labels and converting them to placeholders
 */
export const remarkCodeBlocksToPlaceholders: Plugin = function () {
  return function transformer(tree) {
    if (tree.type === 'root') {
      const root = tree as Root
      root.children = processCodeBlocksWithLabel(root)
    }
    return tree
  }
}

/**
 * Process code blocks with labels and convert them to placeholders
 * Preserves any sourceDoc nodes that have been extracted by remarkPlaceholderParser
 */
function processCodeBlocksWithLabel(ast: Root): RootContent[] {
  const newChildren: RootContent[] = []
  for (let i = 0; i < ast.children.length; i++) {
    const node = ast.children[i]

    if (node.type !== 'code') {
      newChildren.push(node)
      continue
    }

    const codeNode = node as Code
    const metas = parseCodeBlockMeta(codeNode.meta)

    if (!metas['label']) {
      newChildren.push(node)
      continue
    }

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
        finalCommandText = 'changes'
        const objNode: PlaceholderNode = {
          type: 'placeholder',
          placeholderType: 'contextCommand',
          attributes: { command: 'changes' }
        }
        placeholderNode = objNode as unknown as RootContent
        break
      case 'file':
        if (metas['object']) {
          try {
            placeholderNode = createPlaceholderNode(
              'file',
              JSON.stringify(metas['object'])
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
              'symbol',
              JSON.stringify(metas['object'])
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
        ],
        position: {
          start: prevNode.position?.start || {
            line: 0,
            column: 0,
            offset: 0
          },
          end: nextNode.position?.end || { line: 0, column: 0, offset: 0 }
        }
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
        ],
        position: {
          start: node.position?.start || { line: 0, column: 0, offset: 0 },
          end: nextNode.position?.end || { line: 0, column: 0, offset: 0 }
        }
      } as RootContent)
    } else if (
      prevNode &&
      prevNode.type === 'paragraph' &&
      isPrevNodeSameLine
    ) {
      if (Array.isArray(prevNode.children)) {
        prevNode.children.push(
          placeholderNode || { type: 'text', value: ` ${finalCommandText}` }
        )
      }
      if (prevNode.position && node.position) {
        prevNode.position.end = node.position.end
      }
    } else {
      newChildren.push({
        type: 'paragraph',
        children: [
          placeholderNode || { type: 'text', value: finalCommandText }
        ],
        position: node.position
      } as RootContent)
    }
  }
  return newChildren
}
