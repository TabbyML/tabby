import './styles.css'

import React, { useEffect, useState } from 'react'
import Document from '@tiptap/extension-document'
import HardBreak from '@tiptap/extension-hard-break'
import Mention from '@tiptap/extension-mention'
import Paragraph from '@tiptap/extension-paragraph'
import Placeholder from '@tiptap/extension-placeholder'
import Text from '@tiptap/extension-text'
import {
  Editor,
  EditorContent,
  Extension,
  ReactRenderer,
  useEditor
} from '@tiptap/react'
import tippy, { Instance } from 'tippy.js'

import { useLatest } from '@/lib/hooks/use-latest'

import MentionList from './mention-list'
import { SourceOptionItem } from './types'
import { getMentionsWithIndices } from './utils'

const DisableEnter = (onSubmit: Function) =>
  Extension.create({
    addKeyboardShortcuts() {
      return {
        Enter: ({ editor }) => {
          onSubmit(editor)
          return true
        }
      }
    }
  })

interface PromptEditorProps {
  editable: boolean
  content?: string
}

interface MentionContextValue {
  list?: SourceOptionItem[]
  pending: boolean
}

export const MentionContext = React.createContext<MentionContextValue>(
  {} as MentionContextValue
)

const doc_options: SourceOptionItem[] = [
  {
    type: 'source',
    kind: 'doc',
    label: 'tabby',
    id: 'tabbyDoc'
  },
  {
    type: 'source',
    kind: 'doc',
    label: 'skypilot',
    id: 'skypilot'
  }
]

const code_options: SourceOptionItem[] = [
  {
    type: 'source',
    kind: 'code',
    label: 'https://github.com/tabbyml/tabby',
    id: 'tabbyCode'
  },
  {
    type: 'source',
    kind: 'code',
    label: 'https://github.com/facebook/react',
    id: 'react'
  }
]

export const PromptEditor: React.FC<PromptEditorProps> = ({
  editable,
  content
}) => {
  const [items, setItems] = React.useState<SourceOptionItem[]>([])
  const [pending, setPending] = useState(true)

  useEffect(() => {
    setTimeout(() => {
      setItems([...doc_options, ...code_options])
      setPending(false)
    }, 5000)
  }, [])

  const handleSubmit = useLatest((editor: Editor) => {
    const text = editor.getText()
    if (!text) return

    console.log(text)
    console.log(getMentionsWithIndices(editor))
    console.log('submit')
  })

  const onSubmit = (editor: Editor) => {
    handleSubmit.current(editor)
  }

  const editor = useEditor({
    extensions: [
      Document,
      Paragraph,
      Text,
      HardBreak,
      Placeholder.configure({
        placeholder: 'Ask anything...'
      }),
      DisableEnter(onSubmit),
      Mention.configure({
        HTMLAttributes: {
          class: 'mention'
        },
        suggestion: {
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
      })
    ],
    editorProps: {
      attributes: {
        class:
          'prose dark:prose-invert prose-p:my-0 focus:outline-none max-w-none max-h-38 pt-5'
      },
      handleDOMEvents: {
        keydown: (_, event) => {
          if (event.key === 'Enter' && event.shiftKey) {
            if (editor) {
              editor.commands.setHardBreak()
            }
            event.preventDefault()
            return true
          }
        }
      }
    },
    content,
    editable
    // onUpdate(props) {
    //     console.log('upda',props)
    // },
  })

  if (!editor) {
    return null
  }

  return (
    <MentionContext.Provider
      value={{
        list: items,
        pending
      }}
    >
      <div className="text-area-autosize pr-1 max-h-36 overflow-y-auto">
        <EditorContent editor={editor} />
      </div>
    </MentionContext.Provider>
  )
}
