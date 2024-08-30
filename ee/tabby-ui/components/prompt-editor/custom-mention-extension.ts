import { default as MentionExtension } from '@tiptap/extension-mention'
import { mergeAttributes, ReactNodeViewRenderer } from '@tiptap/react'

import { Mention } from './mention'

export const CustomMention = MentionExtension.extend({
  addNodeView() {
    return ReactNodeViewRenderer(Mention)
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
  }
})
