import { default as MentionExtension } from '@tiptap/extension-mention'
import { mergeAttributes, ReactNodeViewRenderer } from '@tiptap/react'

import { MentionForNodeView } from '@/components/mention-tag'

export const CustomMention = MentionExtension.extend({
  addNodeView() {
    return ReactNodeViewRenderer(MentionForNodeView)
  },
  parseHTML() {
    return [
      {
        tag: 'mention'
      }
    ]
  },
  renderHTML({ HTMLAttributes }) {
    return ['mention', mergeAttributes(HTMLAttributes)]
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
