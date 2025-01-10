import React, {
  ForwardedRef,
  useContext,
  useImperativeHandle,
  useRef,
  useState
} from 'react'
import Document from '@tiptap/extension-document'
import Paragraph from '@tiptap/extension-paragraph'
import Placeholder from '@tiptap/extension-placeholder'
import Text from '@tiptap/extension-text'
import { EditorContent, useEditor } from '@tiptap/react'

import { Button } from '@/components/ui/button'
import { IconArrowElbow, IconEdit } from '@/components/ui/icons'
import { Popover, PopoverAnchor, PopoverContent } from '@/components/ui/popover'

import { ChatContext } from './chat'
import { MentionState, PromptFormRef, PromptProps } from './form-editor/types'
import {
  fileItemToSourceItem,
  MentionList,
  PromptFormMentionExtension
} from './form-editor/utils'

/**
 * PromptFormRenderer is the internal component used by React.forwardRef
 * It provides the main logic for the chat input with mention functionality.
 */
function PromptFormRenderer(
  { onSubmit, isLoading }: PromptProps,
  ref: ForwardedRef<PromptFormRef>
) {
  // A ref to track if the mention popover is open (for handling special key events)
  const popoverOpenRef = useRef(false)
  // State controlling the popover
  const [popoverOpen, setPopoverOpen] = useState(false)
  // The popover position is updated to follow the current mention trigger
  const [anchorPos, setAnchorPos] = useState({ left: 0, top: 0 })
  // A ref to store the plain text from the editor
  const inputRef = useRef('')

  // Use forceUpdate to re-render manually when mention state changes
  const [_, forceUpdate] = useState(0)
  const triggerUpdate = () => {
    forceUpdate(i => i + 1)
  }

  // A mention state object to store mention items, command, query, etc.
  const mentionStateRef = useRef<MentionState>({
    items: [],
    command: null,
    query: '',
    selectedIndex: 0
  })

  // If you need a DOM anchor for the popover, you can store it here
  const anchorRef = useRef<HTMLDivElement>(null)

  // Access custom context (e.g., to fetch file suggestions)
  const { listFileInWorkspace } = useContext(ChatContext)

  // Control the popover open/close state
  const handlePopoverChange = (open: boolean) => {
    popoverOpenRef.current = open
    setPopoverOpen(open)
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
        PromptFormMentionExtension.configure({
          // Customize how mention suggestions are fetched and rendered
          suggestion: {
            char: '@', // Trigger character for mention
            items: async ({ query }) => {
              if (!listFileInWorkspace) return []
              const files = await listFileInWorkspace({ query })
              return files?.map(fileItemToSourceItem) || []
            },
            render: () => ({
              onStart: props => {
                const { editor } = props
                const { from } = props.range
                // Calculate the popover position relative to the editor
                const currentLine = editor.view.coordsAtPos(from)
                const editorDom = editor.view.dom.getBoundingClientRect()

                setAnchorPos({
                  left: currentLine.left - editorDom.left,
                  top: currentLine.top - editorDom.top - 10
                })

                // Update mention state
                mentionStateRef.current.items = props.items || []
                mentionStateRef.current.command = props.command
                  ? (attrs: any) => {
                      props.command(attrs)
                      requestAnimationFrame(() => {
                        editor.commands.focus()
                      })
                    }
                  : null
                mentionStateRef.current.query = props.query || ''
                mentionStateRef.current.selectedIndex = 0

                // Open popover and re-render
                setPopoverOpen(true)
                popoverOpenRef.current = true
                triggerUpdate()
                editor.commands.focus()
              },
              onUpdate: props => {
                const { editor } = props
                const { from } = props.range
                const currentLine = editor.view.coordsAtPos(from)
                const editorDom = editor.view.dom.getBoundingClientRect()

                setAnchorPos({
                  left: currentLine.left - editorDom.left,
                  top: currentLine.top - editorDom.top - 10
                })

                // Update mention items, query, etc.
                mentionStateRef.current.items = props.items || []
                mentionStateRef.current.command = props.command
                  ? (attrs: any) => {
                      props.command(attrs)
                      requestAnimationFrame(() => {
                        editor.commands.focus()
                      })
                    }
                  : null
                mentionStateRef.current.query = props.query || ''
                mentionStateRef.current.selectedIndex = 0

                triggerUpdate()
              },
              onExit: () => {
                setPopoverOpen(false)
                popoverOpenRef.current = false
                mentionStateRef.current.command = null
                triggerUpdate()
              },
              onKeyDown: ({ event }) => {
                // Esc key -> close popover
                if (event.key === 'Escape') {
                  setPopoverOpen(false)
                  triggerUpdate()
                  return true
                }

                // Down arrow -> move selection down
                if (event.key === 'ArrowDown') {
                  event.preventDefault()
                  if (!mentionStateRef.current.items.length) return true
                  mentionStateRef.current.selectedIndex =
                    (mentionStateRef.current.selectedIndex + 1) %
                    mentionStateRef.current.items.length
                  triggerUpdate()
                  return true
                }

                // Up arrow -> move selection up
                if (event.key === 'ArrowUp') {
                  event.preventDefault()
                  if (!mentionStateRef.current.items.length) return true
                  const prevIdx = mentionStateRef.current.selectedIndex - 1
                  mentionStateRef.current.selectedIndex =
                    prevIdx < 0
                      ? mentionStateRef.current.items.length - 1
                      : prevIdx
                  triggerUpdate()
                  return true
                }

                // Enter -> confirm selection
                if (event.key === 'Enter') {
                  const { items, selectedIndex, command } =
                    mentionStateRef.current
                  const item = items[selectedIndex]
                  if (item && command) {
                    command({
                      id: `${item.name}-${item.filepath}`,
                      name: item.name,
                      category: 'file',
                      fileItem: item.fileItem
                    })
                  }
                  return true
                }

                return false
              }
            })
          }
        })
      ],
      // On every editor update, store the raw text into inputRef
      onUpdate: ({ editor }) => {
        inputRef.current = editor.getText()
      },
      // Additional editor props (e.g., handleKeyDown to submit on Enter)
      editorProps: {
        handleKeyDown(view, event) {
          // If mention popover is open, let mention extension handle the keys
          if (popoverOpenRef.current) {
            return false
          }
          // Otherwise, handle Enter (without shift) as a submit
          if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault()
            const text = inputRef.current
            // Clear the editor content
            view.dispatch(view.state.tr.delete(0, view.state.doc.content.size))
            inputRef.current = ''
            handleSubmit(undefined, text)
            return true
          }
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
      focus: () => editor?.commands.focus('end'),
      setInput: value => editor?.commands.setContent(value),
      input
    }),
    [editor, input]
  )

  /**
   * A helper function to handle form submission
   */
  const handleSubmit = async (e?: React.FormEvent, text?: string) => {
    e?.preventDefault()
    const content = text ?? editor?.getText()
    if (isLoading || !content?.trim()) return
    await onSubmit(content)
    // Clear editor after successful submit
    editor?.commands.setContent('')
    inputRef.current = ''
  }

  // This element is used as an anchor for the mention popover
  const anchorElement = (
    <div
      ref={anchorRef}
      style={{
        position: 'absolute',
        left: anchorPos.left,
        top: anchorPos.top,
        width: 0,
        height: 0
      }}
    />
  )

  return (
    <form onSubmit={handleSubmit} className="relative">
      <div className="relative flex flex-col px-4">
        {/* Editor & Submit row */}
        <div className="relative flex items-center gap-2">
          <IconEdit className="h-4 w-4 text-muted-foreground" />
          <div className="min-w-0 flex-1">
            {/* TipTap editor content */}
            <EditorContent
              editor={editor}
              className="prose overflow-hidden break-words text-white focus:outline-none"
            />
          </div>
          {anchorElement}
          {/* Submit Button */}
          <Button
            type="submit"
            size="icon"
            disabled={isLoading || input === ''}
          >
            <IconArrowElbow className="h-4 w-4" />
          </Button>
        </div>

        {/* Mention popover (dropdown) */}
        <Popover
          open={popoverOpen}
          modal={false}
          onOpenChange={handlePopoverChange}
        >
          <PopoverAnchor asChild>{anchorElement}</PopoverAnchor>
          <PopoverContent
            className="w-[100%] max-w-none overflow-auto p-0"
            align="start"
            side="top"
            sideOffset={5}
            avoidCollisions
            // Prevent the focus from shifting to the popover
            onOpenAutoFocus={e => e.preventDefault()}
            onCloseAutoFocus={e => e.preventDefault()}
            onMouseDown={e => {
              // Keep focus in the editor
              e.preventDefault()
            }}
          >
            <MentionList
              items={mentionStateRef.current.items}
              command={mentionStateRef.current.command}
              selectedIndex={mentionStateRef.current.selectedIndex}
              onHover={index => {
                mentionStateRef.current.selectedIndex = index
                triggerUpdate()
              }}
            />
          </PopoverContent>
        </Popover>
      </div>
    </form>
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
