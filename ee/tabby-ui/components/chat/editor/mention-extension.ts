// mention-extension.ts
import Mention from '@tiptap/extension-mention'
import { ReactNodeViewRenderer } from '@tiptap/react'

import { MentionComponent } from './mention-component'

export const MENTION_EXTENSION_NAME = 'mention'

export const MentionExtension = Mention.extend({
  addNodeView() {
    return ReactNodeViewRenderer(MentionComponent)
  },

  addAttributes() {
    return {
      id: {
        default: null,
        parseHTML: element => element.getAttribute('data-id'),
        renderHTML: attributes => ({
          'data-id': attributes.id
        })
      },
      label: {
        default: null,
        parseHTML: element => element.getAttribute('data-label'),
        renderHTML: attributes => ({
          'data-label': attributes.label
        })
      },
      category: {
        default: 'file',
        parseHTML: element => element.getAttribute('data-category'),
        renderHTML: attributes => ({
          'data-category': attributes.category
        })
      }
    }
  }
})
