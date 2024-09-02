import { MentionOptions } from '@tiptap/extension-mention'
import { Editor, JSONContent, ReactRenderer } from '@tiptap/react'
import tippy, { GetReferenceClientRect, Instance, Placement } from 'tippy.js'

import { MentionAttributes } from '@/lib/types'

import MentionList, {
  MentionListActions,
  MetionListProps
} from './mention-list'

import 'tippy.js/animations/shift-away.css'

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
  placement?: Placement
}) => MentionOptions['suggestion'] = ({ placement }) => ({
  render: () => {
    let component: ReactRenderer<MentionListActions, MetionListProps>
    let popup: Instance[]

    return {
      onStart: props => {
        // get existing mentions
        const mentions = getMentionsFromEditor(props.editor)

        component = new ReactRenderer(MentionList, {
          props: { ...props, mentions },
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
  }
})

export default suggestion
