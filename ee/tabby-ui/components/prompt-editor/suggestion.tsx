import { MentionOptions } from '@tiptap/extension-mention'
import { ReactRenderer } from '@tiptap/react'
import tippy, { GetReferenceClientRect, Instance, Placement } from 'tippy.js'

import MentionList, {
  MentionListActions,
  MetionListProps
} from './mention-list'
import { getMentionsWithIndices } from './utils'

import 'tippy.js/animations/shift-away.css'

const suggestion: (options: {
  placement?: Placement
}) => MentionOptions['suggestion'] = ({ placement }) => ({
  render: () => {
    let component: ReactRenderer<MentionListActions, MetionListProps>
    let popup: Instance[]

    return {
      onStart: props => {
        // get existing mentions
        const mentions = getMentionsWithIndices(props.editor)

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
