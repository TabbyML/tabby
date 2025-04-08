import type { Root } from 'hast'
import type { Raw } from 'react-markdown/lib/ast-to-react'
import { visit } from 'unist-util-visit'

function createTagFilterExpression(tagNames: string[]): RegExp {
  const escapedTags = tagNames
    .map(tag => tag.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))
    .join('|')
  return new RegExp(
    `<(/?)(?!/?(${escapedTags}))([^>]*)(?=[\\t\\n\\f\\r />])`,
    'gi'
  )
}

/**
 * Escape HTML tags that are not in tagNames
 */
export function customStripTagsPlugin({ tagNames }: { tagNames: string[] }) {
  // const tagFilterExpression = /<(\/?)(?!\/?(think))([^>]*)(?=[\t\n\f\r />])/gi
  const tagFilterExpression = createTagFilterExpression(tagNames)

  return function (tree: Root) {
    visit(tree, 'raw', (node: Raw) => {
      node.value = node.value.replace(tagFilterExpression, '&lt;$2$3')
    })
    return tree
  }
}
