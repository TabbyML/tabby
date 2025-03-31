import { MARKDOWN_CUSTOM_TAGS } from '@/lib/constants'
import type { Root } from 'hast'
import type { Raw } from 'react-markdown/lib/ast-to-react'
import { visit } from 'unist-util-visit'

/**
 * Escape HTML tags that are not in MARKDOWN_CUSTOM_TAGS
 */
// const tagFilterExpression = /<(\/?)(?!\/?(think))([^>]*)(?=[\t\n\f\r />])/gi
const tagFilterExpression = createTagFilterExpression(MARKDOWN_CUSTOM_TAGS)

function createTagFilterExpression(tagNames: typeof MARKDOWN_CUSTOM_TAGS): RegExp {
  const escapedTags = tagNames.map(tag => tag.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|')
  return new RegExp(`<(/?)(?!/?(${escapedTags}))([^>]*)(?=[\\t\\n\\f\\r />])`, 'gi')
}

export function customStripTagsPlugin() {
  return function (tree: Root) {
    visit(tree, 'raw', (node: Raw) => {
      node.value = node.value.replace(tagFilterExpression, '&lt;$2$3')
    })
    return tree
  }
}