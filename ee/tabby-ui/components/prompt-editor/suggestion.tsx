import { MentionOptions } from '@tiptap/extension-mention'
import { ReactRenderer } from '@tiptap/react'
import tippy, { Instance } from 'tippy.js'

import MentionList from './mention-list'

const suggestion = (items: any): MentionOptions['suggestion'] => {
  return {
    items,
    render: () => {
      let component: ReactRenderer
      let popup: Instance

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
            getReferenceClientRect: props.clientRect,
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

          // FIXME
          popup[0].setProps({
            getReferenceClientRect: props.clientRect
          })
        },

        onKeyDown(props) {
          if (props.event.key === 'Escape') {
            // FIXME
            popup[0].hide()

            return true
          }
          // FIXME type check
          return component.ref?.onKeyDown(props)
        },

        onExit() {
          popup[0].destroy()
          component.destroy()
        }
      }
    }
  }
}

export default suggestion
