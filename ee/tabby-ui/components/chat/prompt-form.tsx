import React, {
  ForwardedRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useRef,
  useState
} from 'react'
import { Editor, Extension } from '@tiptap/core'
import Document from '@tiptap/extension-document'
import Paragraph from '@tiptap/extension-paragraph'
import Placeholder from '@tiptap/extension-placeholder'
import Text from '@tiptap/extension-text'
import { EditorContent, useEditor } from '@tiptap/react'

import { useEnterSubmit } from '@/lib/hooks/use-enter-submit'
import { Button, buttonVariants } from '@/components/ui/button'
import { IconArrowElbow, IconEdit } from '@/components/ui/icons'
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger
} from '@/components/ui/tooltip'

import { Popover, PopoverContent } from '../ui/popover'
import { ChatContext } from './chat'
import { PopoverMentionList } from './popover-mention-list'
import {
  CATEGORIES_MENU,
  MentionNodeAttrs,
  MenuState,
  SourceItem,
  SuggestionState
} from './prompt-form-editor/types'

import './prompt-form.css'

import { cn } from '@/lib/utils'

import { PromptFormMentionExtension } from './prompt-form-editor/mention-extension'
import {
  atInfoToSourceItem,
  sourceItemToMentionNodeAttrs
} from './prompt-form-editor/utils'

export interface PromptProps {
  onSubmit: (value: string) => Promise<void>
  isLoading: boolean
  isInitializing?: boolean
}

export interface PromptFormRef {
  focus: () => void
  setInput: (value: string) => void
  input: string
}

function PromptFormRenderer(
  { onSubmit, isLoading, isInitializing }: PromptProps,
  ref: ForwardedRef<PromptFormRef>
) {
  const { formRef } = useEnterSubmit()

  const { provideFileAtInfo } = React.useContext(ChatContext)

  const popoverRef = useRef<HTMLDivElement>(null)
  const selectedItemRef = useRef<HTMLButtonElement>(null)

  const [suggestionState, setSuggestionState] =
    useState<SuggestionState | null>(null)

  const suggestionRef = useRef<Omit<SuggestionState, 'clientRect'> | null>(null)

  const [menuState, setMenuState] = useState<MenuState>({ view: 'categories' })
  const menuStateRef = useRef<MenuState>({ view: 'categories' })

  useEffect(() => {
    menuStateRef.current = menuState
  }, [menuState])

  const updateSelectedIndex = useCallback((index: number) => {
    setSuggestionState(prev =>
      prev ? { ...prev, selectedIndex: index } : null
    )
  }, [])

  /**
   * Scroll the suggestion container so that the selected item is visible.
   */
  const scrollToSelected = useCallback(
    (containerEl: HTMLElement | null, selectedEl: HTMLElement | null) => {
      if (!containerEl || !selectedEl) return
      const containerRect = containerEl.getBoundingClientRect()
      const selectedRect = selectedEl.getBoundingClientRect()

      if (selectedRect.bottom > containerRect.bottom) {
        containerEl.scrollTop += selectedRect.bottom - containerRect.bottom
      } else if (selectedRect.top < containerRect.top) {
        containerEl.scrollTop -= containerRect.top - selectedRect.top
      }
    },
    []
  )

  // Whenever the selected index changes, ensure it's scrolled into view
  useEffect(() => {
    if (suggestionState?.selectedIndex !== undefined) {
      scrollToSelected(popoverRef.current, selectedItemRef.current)
    }
  }, [suggestionState?.selectedIndex, scrollToSelected])

  /**
   * Custom Tiptap extension to handle special keyboard shortcuts.
   */
  const CustomKeyboardShortcuts = Extension.create({
    addKeyboardShortcuts() {
      return {
        'Shift-Enter': () =>
          this.editor.commands.first(({ commands }) => [
            () => commands.newlineInCode(),
            () => commands.createParagraphNear(),
            () => commands.liftEmptyBlock(),
            () => commands.splitBlock()
          ]),
        Enter: ({ editor }) => {
          // If there's an active suggestion or if we are not in the main menu, do not submit
          if (
            suggestionRef.current ||
            menuStateRef.current.view !== 'categories'
          ) {
            return false
          }
          handleSubmit(undefined, editor.getText())
          editor.commands.setContent('')
          return true
        }
      }
    }
  })

  /**
   * Handle item selection when '@' mention is triggered.
   */
  const handleItemSelection = useCallback(
    (item: SourceItem, command?: (props: MentionNodeAttrs) => void) => {
      // If user is in the main menu (categories) and picks something like "files"
      if (menuStateRef.current.view === 'categories') {
        switch (item.name.toLowerCase()) {
          case 'files': {
            const state: MenuState = { view: 'files' }
            setMenuState(state)
            menuStateRef.current = state

            // If a function is provided to fetch file info, get them and update suggestions
            if (provideFileAtInfo) {
              provideFileAtInfo().then(files => {
                if (!files) return
                const items = files.map(atInfoToSourceItem)
                setSuggestionState(prev =>
                  prev
                    ? {
                        ...prev,
                        items,
                        selectedIndex: 0
                      }
                    : null
                )
                suggestionRef.current = {
                  items,
                  selectedIndex: 0,
                  command: command || suggestionRef.current?.command!
                }
              })
            }
            break
          }
          case 'symbols':
            // TODO: Implement symbol selection if needed
            break
          default:
            break
        }
      } else {
        const attrs = sourceItemToMentionNodeAttrs(item)
        if (command) {
          command(attrs)
        } else if (suggestionRef.current?.command) {
          suggestionRef.current.command(attrs)
        }
      }
    },
    [provideFileAtInfo]
  )

  /**
   * Handle up/down/enter keys in the suggestion popover.
   */
  const handleKeyDown = useCallback(
    (event: KeyboardEvent) => {
      const currentSuggestion = suggestionRef.current
      if (!currentSuggestion || !currentSuggestion.items?.length) return false

      switch (event.key) {
        case 'ArrowUp':
        case 'ArrowDown': {
          event.preventDefault()
          const direction = event.key === 'ArrowUp' ? -1 : 1

          setSuggestionState(prev => {
            if (!prev) return null
            const length = currentSuggestion.items.length
            const newIndex = (prev.selectedIndex + direction + length) % length
            return { ...prev, selectedIndex: newIndex }
          })

          suggestionRef.current = {
            ...currentSuggestion,
            selectedIndex:
              (currentSuggestion.selectedIndex +
                direction +
                currentSuggestion.items.length) %
              currentSuggestion.items.length
          }
          return true
        }
        case 'Enter': {
          event.preventDefault()
          const selectedItem =
            currentSuggestion.items[currentSuggestion.selectedIndex]
          if (selectedItem) {
            handleItemSelection(selectedItem, currentSuggestion.command)
          }
          return true
        }
        default:
          return false
      }
    },
    [handleItemSelection]
  )

  /**
   * Initialize the Tiptap editor with the specified extensions and placeholder.
   */
  const editor = useEditor({
    extensions: [
      Document,
      Paragraph,
      Text,
      CustomKeyboardShortcuts,
      Placeholder.configure({
        showOnlyWhenEditable: true,
        placeholder: 'Ask anything...'
      }),
      PromptFormMentionExtension.configure({
        HTMLAttributes: {
          class: 'mention'
        },
        suggestion: {
          char: '@',
          allowSpaces: true,
          items: async ({ query }): Promise<SourceItem[]> => {
            if (!provideFileAtInfo) return []

            if (menuStateRef.current.view === 'categories') {
              return CATEGORIES_MENU
            }

            try {
              const files = await provideFileAtInfo({ query })
              return files?.map(atInfoToSourceItem) || []
            } catch (error) {
              // TODO: handle or log error if needed
              return []
            }
          },
          render: () => ({
            onStart: props => {
              const newState: SuggestionState = {
                items: props.items as SourceItem[],
                command: props.command,
                clientRect: props.clientRect!,
                selectedIndex: 0
              }
              suggestionRef.current = {
                items: props.items as SourceItem[],
                command: props.command,
                selectedIndex: 0
              }
              setSuggestionState(newState)

              // Ensure the editor keeps focus after mention starts
              requestAnimationFrame(() => {
                editor?.commands.focus()
              })
            },
            onUpdate: props => {
              const newState: SuggestionState = {
                items: props.items as SourceItem[],
                command: props.command,
                clientRect: props.clientRect!,
                selectedIndex: 0
              }
              suggestionRef.current = {
                items: props.items as SourceItem[],
                command: props.command,
                selectedIndex: 0
              }
              setSuggestionState(newState)

              requestAnimationFrame(() => {
                editor?.commands.focus()
              })
            },
            onKeyDown: ({ event }) => {
              if (['ArrowUp', 'ArrowDown', 'Enter'].includes(event.key)) {
                return handleKeyDown(event)
              }
              return false
            },
            onExit: () => {
              const initialMenuState: MenuState = { view: 'categories' }
              setMenuState(initialMenuState)
              menuStateRef.current = initialMenuState
              suggestionRef.current = null
              setSuggestionState(null)
            }
          })
        }
      })
    ],
    content: ''
  })

  const input = editor?.getText() || ''

  useImperativeHandle(
    ref,
    () => ({
      focus: () => {
        editor?.commands.focus('end')
      },
      setInput: (str: string) => {
        editor?.commands.setContent(str)
      },
      input
    }),
    [editor, input]
  )

  /**
   * Submit handler for the form. Called on Enter or via button click.
   */
  const handleSubmit = async (e?: React.FormEvent, text?: string) => {
    e?.preventDefault()
    if (isLoading) return

    const finalText = text ?? input
    if (!finalText?.trim()) return

    await onSubmit(finalText)
    editor?.commands.setContent('')
  }

  /**
   * Calculate the vertical offset for the Popover, depending on the active menu view.
   */
  const topOffset = (() => {
    if (!suggestionState) return 0
    const { view } = menuStateRef.current
    if (view === 'categories') {
      return 70
    } else if (view === 'files' && suggestionState.items.length > 0) {
      // Each item is ~42px tall, display max 4 items
      return Math.min(suggestionState.items.length * 42, 4 * 42)
    }
    return 70
  })()

  return (
    <>
      <form onSubmit={handleSubmit} ref={formRef}>
        <div className="bg-background relative flex max-h-60 w-full grow flex-col overflow-hidden px-8 sm:rounded-md sm:border sm:px-12">
          <span
            className={cn(
              buttonVariants({ size: 'sm', variant: 'ghost' }),
              'bg-background hover:bg-background absolute left-0 top-4 h-8 w-8 rounded-full p-0 sm:left-4'
            )}
          >
            <IconEdit />
          </span>

          <div className="min-h-[60px] w-full resize-none bg-transparent py-[1.3rem] focus-within:outline-none sm:pl-4">
            <EditorContent
              editor={editor}
              className="prose dark:prose-invert prose-p:my-0 focus:outline-none"
            />
          </div>

          <div className="absolute right-0 top-4 sm:right-4">
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  type="submit"
                  size="icon"
                  disabled={isInitializing || isLoading || input === ''}
                >
                  <IconArrowElbow />
                  <span className="sr-only">Send message</span>
                </Button>
              </TooltipTrigger>
              <TooltipContent>Send message</TooltipContent>
            </Tooltip>
          </div>
        </div>
      </form>

      {/* Suggestion popover, triggered if mentionState is not null */}
      {suggestionState && (
        <Popover open={true} modal={false}>
          <PopoverContent
            ref={popoverRef}
            className="p-0 w-[280px] overflow-y-auto"
            style={{
              position: 'absolute',
              left: suggestionState.clientRect()?.left ?? 0,
              top: (suggestionState.clientRect()?.top ?? 0) - topOffset,
              maxHeight: '200px'
            }}
            align="start"
            onOpenAutoFocus={e => e.preventDefault()}
            onPointerDownOutside={e => e.preventDefault()}
            onFocusOutside={e => e.preventDefault()}
          >
            <PopoverMentionList
              items={suggestionState.items}
              selectedIndex={suggestionState.selectedIndex}
              onUpdateSelectedIndex={updateSelectedIndex}
              handleItemSelection={item =>
                handleItemSelection(item, suggestionRef.current?.command)
              }
            />
          </PopoverContent>
        </Popover>
      )}
    </>
  )
}

/**
 * Export PromptForm with a forwarded ref to access internal methods.
 */
export const PromptForm = React.forwardRef<PromptFormRef, PromptProps>(
  PromptFormRenderer
)

export default PromptForm
