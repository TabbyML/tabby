import 'tippy.js/animations/shift-away.css'

import { MentionNodeAttrs, MentionOptions } from '@tiptap/extension-mention'
import { PluginKey } from '@tiptap/pm/state'
import { Content, Editor, JSONContent, ReactRenderer } from '@tiptap/react'
import tippy, { GetReferenceClientRect, Instance, Placement } from 'tippy.js'

import { MentionAttributes } from '@/lib/types'
import { isCodeSourceContext } from '@/lib/utils'

import { MENTION_EXTENSION_NAME } from './mention-extension'
import MentionList, {
  MentionListActions,
  MetionListProps
} from './mention-list'

const getMentionsFromEditor = (editor: Editor) => {
  const json = editor.getJSON()
  const mentions: MentionAttributes[] = []
  let textLength = 0

  const traverse = (node: JSONContent) => {
    if (node.type === 'text') {
      textLength += node?.text?.length || 0
    } else if (node.type === 'mention') {
      if (node?.attrs?.id) {
        mentions.push({
          id: node.attrs.id,
          label: node.attrs.label,
          kind: node.attrs.kind
        })
      }
    }

    if (node.content) {
      node.content.forEach(traverse)
    }
  }

  traverse(json)
  return mentions
}

const suggestion: (options: {
  disabled?: boolean
  category: 'doc' | 'code'
  placement?: Placement
  pluginKey: PluginKey
  char?: string
}) => MentionOptions['suggestion'] = ({
  disabled,
  category,
  placement,
  char = '@',
  pluginKey
}) => ({
  render: () => {
    let component: ReactRenderer<MentionListActions, MetionListProps>
    let popup: Instance[]

    return {
      onStart: props => {
        // get existing mentions
        const mentions = getMentionsFromEditor(props.editor)

        component = new ReactRenderer(MentionList, {
          props: { ...props, mentions, category },
          editor: props.editor
        })

        if (!props.clientRect) {
          return
        }

        popup = tippy('body', {
          getReferenceClientRect: props.clientRect as GetReferenceClientRect,
          appendTo: () => document.body,
          content: component.element,
          showOnCreate: true,
          interactive: true,
          trigger: 'manual',
          placement: placement || 'bottom-start',
          animation: 'shift-away',
          maxWidth: '400px'
        })
      },
      onUpdate(props) {
        // called on query change
        component.updateProps(props)

        if (!props.clientRect) {
          return
        }

        popup[0].setProps({
          getReferenceClientRect: props.clientRect as GetReferenceClientRect
        })
      },

      onKeyDown(props) {
        if (props.event.key === 'Escape') {
          popup[0].hide()

          return true
        }
        return component.ref?.onKeyDown(props) ?? false
      },

      onExit() {
        popup[0].destroy()
        component.destroy()
      }
    }
  },
  char,
  pluginKey,
  command: ({ editor, range, props }) => {
    if (category === 'code') {
      insertCodebaseMention(editor, range, props)
      return
    }

    // increase range.to by one when the next node is of type "text"
    // and starts with a space character
    const nodeAfter = editor.view.state.selection.$to.nodeAfter
    const overrideSpace = nodeAfter?.text?.startsWith(' ')

    if (overrideSpace) {
      range.to += 1
    }

    editor
      .chain()
      .focus()
      .insertContentAt(range, [
        {
          type: MENTION_EXTENSION_NAME,
          attrs: props
        },
        {
          type: 'text',
          text: ' '
        }
      ])
      .run()

    // get reference to `window` object from editor element, to support cross-frame JS usage
    editor.view.dom.ownerDocument.defaultView?.getSelection()?.collapseToEnd()
  },
  allow: ({ state, range }) => {
    if (disabled) {
      return false
    }

    const $from = state.doc.resolve(range.from)
    const type = state.schema.nodes[MENTION_EXTENSION_NAME]
    const allow = !!$from.parent.type.contentMatch.matchType(type)

    return allow
  }
})

function insertCodebaseMention(
  editor: Editor,
  range: { from: number; to: number },
  props: MentionNodeAttrs
) {
  const { doc } = editor.state
  let mentionPos: number | undefined

  // Save the current selection
  const currentSelection = editor.state.selection.$from

  // Traverse the document to find the existing mention
  doc.descendants((node, pos) => {
    if (node.type.name === 'mention' && isCodeSourceContext(node.attrs.kind)) {
      mentionPos = pos
      // Stop traversing once we find the mention
      return false
    }
    return true
  })

  // Delete the triggering character `#`
  editor.chain().deleteRange({ from: range.from, to: range.to }).run()

  const isExistingCodeMentionNode = mentionPos !== undefined
  // Check if the first node is a paragraph
  const firstNode = doc.firstChild

  const content: Content = isExistingCodeMentionNode
    ? [
        {
          type: MENTION_EXTENSION_NAME,
          attrs: props
        }
      ]
    : [
        {
          type: MENTION_EXTENSION_NAME,
          attrs: props
        },
        {
          type: 'text',
          text: ' '
        }
      ]
  if (firstNode && firstNode.type.name === 'paragraph') {
    // If the first child of the paragraph is a mention, replace it
    const firstChild = firstNode.firstChild
    if (
      firstChild &&
      firstChild.type.name === 'mention' &&
      isCodeSourceContext(firstChild.attrs.kind)
    ) {
      const mentionNodeSize = firstChild.nodeSize
      editor
        .chain()
        .deleteRange({ from: 1, to: 1 + mentionNodeSize })
        .insertContentAt(1, content)
        .focus()
        .run()
    } else {
      // Insert the new mention as the first child of the paragraph
      editor.chain().insertContentAt(1, content).focus().run()
    }
  } else {
    // If the first node is not a paragraph, insert the new mention at the document start
    editor.chain().insertContentAt(0, content).focus().run()
  }

  editor.commands.focus(
    isExistingCodeMentionNode ? currentSelection.pos : currentSelection.pos + 1
  )
}

export default suggestion
