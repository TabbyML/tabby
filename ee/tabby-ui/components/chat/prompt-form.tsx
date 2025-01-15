import React, { ForwardedRef, useContext, useImperativeHandle } from 'react'
import Document from '@tiptap/extension-document'
import Paragraph from '@tiptap/extension-paragraph'
import Placeholder from '@tiptap/extension-placeholder'
import Text from '@tiptap/extension-text'
import {
  EditorContent,
  Extension,
  ReactRenderer,
  useEditor
} from '@tiptap/react'

import './prompt-form.css'

import tippy, { Instance } from 'tippy.js'

import { NEWLINE_CHARACTER } from '@/lib/constants'
import { useLatest } from '@/lib/hooks/use-latest'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { IconArrowElbow, IconEdit } from '@/components/ui/icons'

// import { Popover, PopoverAnchor, PopoverContent } from '@/components/ui/popover'

import { ChatContext } from './chat'
import {
  MentionList,
  MentionListActions,
  MentionListProps,
  PromptFormMentionExtension
} from './form-editor/mention'
import { PromptFormRef, PromptProps } from './form-editor/types'
import { fileItemToSourceItem } from './form-editor/utils'

/**
 * PromptFormRenderer is the internal component used by React.forwardRef
 * It provides the main logic for the chat input with mention functionality.
 */
function PromptFormRenderer(
  { onSubmit, isLoading }: PromptProps,
  ref: ForwardedRef<PromptFormRef>
) {
  // Access custom context (e.g., to fetch file suggestions)
  const { listFileInWorkspace } = useContext(ChatContext)

  const doSubmit = useLatest(async () => {
    if (isLoading || !editor) return

    const text = editor.getText({ blockSeparator: NEWLINE_CHARACTER }).trim()
    if (!text) return

    const result = onSubmit(text)
    editor?.chain().clearContent().focus().run()

    return result
  })

  const handleSubmit = () => {
    doSubmit.current()
  }

  // Set up the TipTap editor with mention extension
  const editor = useEditor(
    {
      extensions: [
        Document,
        Paragraph,
        Text,
        Placeholder.configure({
          placeholder: 'typing...'
        }),
        CustomKeyboardShortcuts(handleSubmit),
        PromptFormMentionExtension.configure({
          // Customize how mention suggestions are fetched and rendered
          suggestion: {
            char: '@', // Trigger character for mention
            items: async ({ query }) => {
              if (!listFileInWorkspace) return []
              const files = await listFileInWorkspace({ query })
              return files?.map(fileItemToSourceItem) || []
            },
            render: () => {
              let component: ReactRenderer<MentionListActions, MentionListProps>
              let popup: Instance[]

              // const getRemInPixels = () => {
              //   return parseFloat(getComputedStyle(document.documentElement).fontSize);
              // };

              const updatePopperWidth = (instance: Instance) => {
                const targetWidth =
                  instance.reference.getBoundingClientRect().width
                instance.popper.style.maxWidth = `${targetWidth}px`
                // instance.popper.style.maxWidth = `${targetWidth}px`
                // instance.popper.style.width = `calc(${targetWidth}px - 1rem)`
                instance.popper.style.width = `${targetWidth - 16}px`
              }

              const handleResize = () => {
                if (popup && popup[0]) {
                  updatePopperWidth(popup[0])
                }
              }

              return {
                onStart: props => {
                  component = new ReactRenderer(MentionList, {
                    props: { ...props, listFileInWorkspace },
                    editor: props.editor
                  })

                  const container = document.querySelector(
                    '#chat-panel-container'
                  )
                  if (!container) {
                    return
                  }

                  popup = tippy('#chat-panel-container', {
                    // getReferenceClientRect: () => container.getBoundingClientRect(),
                    appendTo: () => document.body,
                    content: component.element,
                    showOnCreate: true,
                    interactive: true,
                    trigger: 'manual',
                    placement: 'top-start',
                    animation: 'shift-away',
                    maxWidth: document.documentElement.clientWidth,
                    offset({ placement, popper, reference }) {
                      return [8, 6]
                    },
                    onCreate(instance) {
                      updatePopperWidth(instance)
                    },
                    onMount() {
                      window.addEventListener('resize', handleResize)
                    },
                    onHidden() {
                      window.removeEventListener('resize', handleResize)
                    }
                  })
                },
                onUpdate: props => {
                  // const { editor } = props
                  // const { from } = props.range
                  // const currentLine = editor.view.coordsAtPos(from)
                  // const editorDom = editor.view.dom.getBoundingClientRect()

                  // setAnchorPos({
                  //   left: currentLine.left - editorDom.left,
                  //   top: currentLine.top - editorDom.top - 10
                  // })

                  // Update mention items, query, etc.
                  // mentionStateRef.current.items = props.items || []
                  // mentionStateRef.current.command = props.command
                  //   ? (attrs: any) => {
                  //     props.command(attrs)
                  //     requestAnimationFrame(() => {
                  //       editor.commands.focus()
                  //     })
                  //   }
                  //   : null
                  // mentionStateRef.current.query = props.query || ''
                  // mentionStateRef.current.selectedIndex = 0
                  component.updateProps(props)
                },
                onExit: () => {
                  popup[0].destroy()
                  component.destroy()
                  // handlePopoverChange(false)
                  // mentionStateRef.current.command = null
                },
                onKeyDown: props => {
                  if (props.event.key === 'Escape') {
                    popup[0].hide()

                    return true
                  }
                  return component.ref?.onKeyDown(props) ?? false
                  // Esc key -> close popover
                  // if (event.key === 'Escape') {
                  //   setPopoverOpen(false)
                  //   return true
                  // }

                  // // Down arrow -> move selection down
                  // if (event.key === 'ArrowDown') {
                  //   event.preventDefault()
                  //   if (!mentionStateRef.current.items.length) return true
                  //   mentionStateRef.current.selectedIndex =
                  //     (mentionStateRef.current.selectedIndex + 1) %
                  //     mentionStateRef.current.items.length
                  //   return true
                  // }

                  // // Up arrow -> move selection up
                  // if (event.key === 'ArrowUp') {
                  //   event.preventDefault()
                  //   if (!mentionStateRef.current.items.length) return true
                  //   const prevIdx = mentionStateRef.current.selectedIndex - 1
                  //   mentionStateRef.current.selectedIndex =
                  //     prevIdx < 0
                  //       ? mentionStateRef.current.items.length - 1
                  //       : prevIdx
                  //   return true
                  // }

                  // // Enter -> confirm selection
                  // if (event.key === 'Enter') {
                  //   const { items, selectedIndex, command } =
                  //     mentionStateRef.current
                  //   const item = items[selectedIndex]
                  //   if (item && command) {
                  //     command({
                  //       category: 'file',
                  //       filepath: item.filepath
                  //     })
                  //   }
                  //   return true
                  // }

                  // return false
                }
              }
            }
          }
        })
      ],
      editorProps: {
        attributes: {
          class: cn(
            'prose max-w-none font-sans dark:prose-invert focus:outline-none prose-p:my-0'
          )
        }
      }
    },
    [listFileInWorkspace]
  )

  // Current text from the editor (for checking if the submit button is disabled)
  const input = editor?.getText() || ''

  /**
   * Expose methods to the parent component via ref
   */
  useImperativeHandle(
    ref,
    () => ({
      focus: () => editor?.commands.focus(),
      setInput: value => editor?.commands.setContent(value),
      input
    }),
    [editor, input]
  )

  return (
    <div className="relative flex flex-col px-2.5">
      {/* Editor & Submit row */}
      <div className="relative flex items-start gap-2">
        <span className="mt-[1.375rem]">
          <IconEdit className="h-4 w-4" />
        </span>
        <div
          className="max-h-32 flex-1 overflow-y-auto py-4"
          onClick={e => {
            if (editor && !editor.isFocused) {
              editor?.commands.focus()
            }
          }}
        >
          {/* TipTap editor content */}
          <EditorContent
            editor={editor}
            className={cn(
              'prose overflow-hidden break-words text-foreground focus:outline-none'
            )}
          />
        </div>
        {/* Submit Button */}
        <Button
          className="mt-4 h-7 w-7"
          size="icon"
          disabled={isLoading || input === ''}
          onClick={handleSubmit}
        >
          <IconArrowElbow className="h-3.5 w-3.5" />
        </Button>
      </div>
    </div>
  )
}

/**
 * Export the PromptForm as a forwardRef component
 */
export const PromptForm = React.forwardRef<PromptFormRef, PromptProps>(
  PromptFormRenderer
)

/**
 * For convenience, also export it as default
 */
export default PromptForm

const CustomKeyboardShortcuts = (onSubmit: () => void) =>
  Extension.create({
    addKeyboardShortcuts() {
      return {
        Enter: ({ editor }) => {
          onSubmit()
          return true
        },
        'Shift-Enter': () => {
          return this.editor.commands.first(({ commands }) => [
            () => commands.newlineInCode(),
            () => commands.createParagraphNear(),
            () => commands.liftEmptyBlock(),
            () => commands.splitBlock()
          ])
        }
      }
    }
  })
