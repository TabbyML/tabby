import Mention from '@tiptap/extension-mention'
import { ReactNodeViewRenderer } from '@tiptap/react'

import { PromptFormMentionComponent } from './mention-component'

export const MENTION_EXTENSION_NAME = 'mention'

export const PromptFormMentionExtension = Mention.extend({
  addNodeView() {
    return ReactNodeViewRenderer(PromptFormMentionComponent)
  },
  renderText({ node }) {
    return `[[atSource:${JSON.stringify(node.attrs.atInfo)}]]`
  },
  addAttributes() {
    return {
      name: {
        default: null,
        parseHTML: element => element.getAttribute('data-name'),
        renderHTML: attributes => ({
          'data-name': attributes.name
        })
      },
      category: {
        default: 'file',
        parseHTML: element => element.getAttribute('data-category'),
        renderHTML: attributes => ({
          'data-category': attributes.category
        })
      },
      atInfo: {
        default: null,
        parseHTML: element => {
          const atInfo = element.getAttribute('data-at-info')
          return atInfo ? JSON.parse(atInfo) : null
        },
        renderHTML: attributes => ({
          'data-at-info': JSON.stringify(attributes.atInfo)
        })
      }
    }
  }
})
