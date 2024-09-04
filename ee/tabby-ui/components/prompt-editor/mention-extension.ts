import Mention from '@tiptap/extension-mention'
import { ReactNodeViewRenderer } from '@tiptap/react'

import { MentionForNodeView } from '@/components/mention-tag'

export const MENTION_EXTENSION_NAME = 'mention'

export const MentionExtension = Mention.extend({
  addNodeView() {
    return ReactNodeViewRenderer(MentionForNodeView)
  },
  renderText({ node }) {
    return `[[source:${node.attrs.id}]]`
  },
  addAttributes() {
    return {
      id: {
        default: null,
        parseHTML: element => element.getAttribute('data-id'),
        renderHTML: attributes => {
          if (!attributes.id) {
            return {}
          }

          return {
            'data-id': attributes.id
          }
        }
      },
      label: {
        default: null,
        parseHTML: element => element.getAttribute('data-label'),
        renderHTML: attributes => {
          if (!attributes.label) {
            return {}
          }

          return {
            'data-label': attributes.label
          }
        }
      },
      kind: {
        default: null,
        parseHTML: element => element.getAttribute('data-kind'),
        renderHTML: attributes => {
          if (!attributes.kind) {
            return {}
          }

          return {
            'data-kind': attributes.kind
          }
        }
      }
    }
  }
})
