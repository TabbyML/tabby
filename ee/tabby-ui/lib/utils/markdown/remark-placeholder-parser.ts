/**
 * Remark plugin for handling source references with syntax [[source:type:id]]
 */

import { Node } from 'mdast'
import { Root } from 'remark-stringify/lib'
import { Plugin } from 'unified'

export interface PlaceholderNode extends Node {
  type: 'placeholder'
  // TODO: this mention type or data source type should be defined as type
  placeholderType: 'source' | 'file' | 'symbol' | 'contextCommand' | 'undefined'
  attributes: Record<string, any>
}

/**
 * Remark plugin for parsing and transforming placeholder references
 * in markdown text with syntax [[type:value]]
 */
export const remarkPlaceholderParser: Plugin = function () {
  return function transformer(tree) {
    return transformPlaceholders(tree as Root)
  }
}

// Export the original name for backward compatibility
export const remarkPlaceHolderSource = remarkPlaceholderParser

export function parsePlaceholder(
  text: string
): { placeholderNode: PlaceholderNode; matchLength: number } | null {
  const match = /^\[\[([^\]]+)\]\]/.exec(text)
  if (!match) return null

  const content = match[1]
  const matchLength = match[0].length

  const colonIndex = content.indexOf(':')
  if (colonIndex === -1) {
    // Generic case with no colon
    return {
      placeholderNode: {
        type: 'placeholder',
        placeholderType: 'undefined',
        attributes: {
          content: text
        }
      },
      matchLength
    }
  }

  const prefix = content.substring(0, colonIndex)
  const restContent = content.substring(colonIndex + 1)

  switch (prefix) {
    case 'source': {
      if (!!restContent) {
        return {
          placeholderNode: {
            type: 'placeholder',
            placeholderType: 'source',
            attributes: {
              sourceId: restContent
            }
          },
          matchLength
        }
      }
    }
    case 'file': {
      const value = restContent
      return {
        placeholderNode: {
          type: 'placeholder',
          placeholderType: 'file',
          attributes: {
            object: value
          }
        },
        matchLength
      }
    }
    case 'symbol': {
      const value = restContent
      return {
        placeholderNode: {
          type: 'placeholder',
          placeholderType: 'symbol',
          attributes: {
            object: value
          }
        },
        matchLength
      }
    }
    case 'contextCommand': {
      const value = restContent
      return {
        placeholderNode: {
          type: 'placeholder',
          placeholderType: 'contextCommand',
          attributes: {
            command: value
          }
        },
        matchLength
      }
    }
    default: {
      return {
        placeholderNode: {
          type: 'placeholder',
          placeholderType: 'undefined',
          attributes: {
            content: match[0]
          }
        },
        matchLength
      }
    }
  }

  return null
}

export function transformPlaceholders(tree: any): any {
  if (!tree) return tree

  if (tree.type === 'root' && Array.isArray(tree.children)) {
    for (let i = 0; i < tree.children.length; i++) {
      const node = tree.children[i]

      if (node.type === 'paragraph' && Array.isArray(node.children)) {
        const newChildren = []
        for (let j = 0; j < node.children.length; j++) {
          const child = node.children[j]
          if (child.type === 'text') {
            let text = child.value
            let position = 0
            let placeholderIndex

            while ((placeholderIndex = text.indexOf('[[', position)) !== -1) {
              if (placeholderIndex > position) {
                newChildren.push({
                  type: 'text',
                  value: text.substring(position, placeholderIndex)
                })
              }

              const placeholderText = text.substring(placeholderIndex)
              const parsed = parsePlaceholder(placeholderText)

              if (parsed) {
                newChildren.push(parsed.placeholderNode)
                position = placeholderIndex + parsed.matchLength
              } else {
                newChildren.push({
                  type: 'text',
                  value: '[['
                })
                position = placeholderIndex + 2
              }
            }

            if (position < text.length) {
              newChildren.push({
                type: 'text',
                value: text.substring(position)
              })
            }
          } else {
            newChildren.push(child)
          }
        }

        node.children = newChildren
      }

      if (
        node.children &&
        Array.isArray(node.children) &&
        node.type !== 'paragraph'
      ) {
        transformPlaceholders({ type: 'root', children: node.children })
      }
    }
  }

  return tree
}

export function placeholderToString(node: PlaceholderNode): string {
  const formatPlaceholder = (content: string): string => {
    return ` ${content} `
  }
  switch (node.placeholderType) {
    case 'source':
      return formatPlaceholder(`[[source:${node.attributes.sourceId}]]`)
    case 'file':
      return formatPlaceholder(`[[file:${node.attributes.object}]]`)
    case 'symbol':
      return formatPlaceholder(`[[symbol:${node.attributes.object}]]`)
    case 'contextCommand':
      return formatPlaceholder(`[[contextCommand:${node.attributes.command}]]`)
    case 'undefined':
    default:
      return node.attributes.content ?? ''
  }
}
