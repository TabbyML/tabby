import { MentionOptions } from '@tiptap/extension-mention'
import { ReactRenderer } from '@tiptap/react'
import tippy, { GetReferenceClientRect, Instance } from 'tippy.js'

import MentionList, {
  MentionListActions,
  MetionListProps
} from './mention-list'

const suggestion: MentionOptions['suggestion'] = {
  render: () => {
    let component: ReactRenderer<MentionListActions, MetionListProps>
    let popup: Instance[]

    return {
      onStart: props => {
        component = new ReactRenderer(MentionList, {
          props,
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
          placement: 'bottom-start'
        })
      },
      onUpdate(props) {
        // call once query change
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
}

export default suggestion
