/* eslint-disable no-console */
import React, { useCallback, useEffect, useRef, useState } from 'react'
import { Extension } from '@tiptap/core'
import Document from '@tiptap/extension-document'
import Paragraph from '@tiptap/extension-paragraph'
import Placeholder from '@tiptap/extension-placeholder'
import Text from '@tiptap/extension-text'
import { EditorContent, useEditor } from '@tiptap/react'
import { UseChatHelpers } from 'ai/react'

import { useEnterSubmit } from '@/lib/hooks/use-enter-submit'
import { Button } from '@/components/ui/button'
import { IconArrowElbow } from '@/components/ui/icons'
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger
} from '@/components/ui/tooltip'

import { Popover, PopoverContent } from '../ui/popover'
import { ChatContext } from './chat'
import { FileList } from './FileList'
import { SuggestionItem } from './types'

import './prompt-form.css'

import { CategoryMenu } from './editor/CategoryMenu'
import { MentionExtension } from './editor/mention-extension'

export interface PromptProps
  extends Pick<UseChatHelpers, 'input' | 'setInput'> {
  onSubmit: (value: string) => Promise<void>
  isLoading: boolean
  chatInputRef: React.RefObject<HTMLTextAreaElement>
  isInitializing?: boolean
}

export interface PromptFormRef {
  focus: () => void
}

type MenuView = 'categories' | 'files' | 'symbols'

interface MenuState {
  view: MenuView
  category?: 'file' | 'symbol'
}

interface SuggestionState {
  items: SuggestionItem[]
  command: (item: {
    id: string
    label: string
    category: 'file' | 'symbol'
  }) => void
  clientRect: () => DOMRect | null
  selectedIndex: number
}

const CustomKeyboardShortcuts = Extension.create({
  addKeyboardShortcuts() {
    return {
      'Shift-Enter': () => {
        console.log('[CustomKeyboardShortcuts] Shift-Enter pressed')
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

function PromptFormRenderer(
  {
    onSubmit,
    input,
    setInput,
    isLoading,
    isInitializing,
    chatInputRef
  }: PromptProps,
  ref: React.ForwardedRef<PromptFormRef>
) {
  const { formRef } = useEnterSubmit()
  const { provideFileAtInfo } = React.useContext(ChatContext)
  const popoverRef = useRef<HTMLDivElement>(null)
  const selectedItemRef = useRef<HTMLButtonElement>(null)

  const [suggestionState, setSuggestionState] =
    useState<SuggestionState | null>(null)
  const suggestionRef = useRef<Omit<
    SuggestionState,
    'clientRect' | 'selectedIndex'
  > | null>(null)

  const [menuState, setMenuState] = useState<MenuState>({ view: 'categories' })

  const categoryItems: {
    label: string
    category: 'file' | 'symbol'
  }[] = [
    { label: 'Files', category: 'file' },
    { label: 'Symbols', category: 'symbol' }
  ]

  const [categorySelectedIndex, setCategorySelectedIndex] = useState(0)

  const updateSelectedIndex = useCallback((index: number) => {
    console.log('[PromptForm] Updating mention suggestion index:', index)
    setSuggestionState(prev =>
      prev ? { ...prev, selectedIndex: index } : null
    )
  }, [])

  const scrollToSelected = useCallback(
    (containerEl: HTMLElement | null, selectedEl: HTMLElement | null) => {
      console.log('[PromptForm] Scrolling to selected element')
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

  useEffect(() => {
    if (suggestionState?.selectedIndex !== undefined) {
      console.log('[PromptForm] Selected index changed, updating scroll')
      scrollToSelected(popoverRef.current, selectedItemRef.current)
    }
  }, [suggestionState?.selectedIndex, scrollToSelected])

  const handleKeyDown = useCallback(
    (event: KeyboardEvent) => {
      console.log('[PromptForm] Key pressed in mention suggestions:', event.key)
      const currentSuggestion = suggestionRef.current
      if (!currentSuggestion?.items?.length) return false

      switch (event.key) {
        case 'ArrowUp': {
          event.preventDefault()
          console.log('[PromptForm] Handling ArrowUp')
          setSuggestionState(prev => {
            if (!prev) return null
            const newIndex =
              prev.selectedIndex > 0
                ? prev.selectedIndex - 1
                : currentSuggestion.items.length - 1
            return { ...prev, selectedIndex: newIndex }
          })
          return true
        }

        case 'ArrowDown': {
          event.preventDefault()
          console.log('[PromptForm] Handling ArrowDown')
          setSuggestionState(prev => {
            if (!prev) return null
            const newIndex =
              prev.selectedIndex < currentSuggestion.items.length - 1
                ? prev.selectedIndex + 1
                : 0
            return { ...prev, selectedIndex: newIndex }
          })
          return true
        }

        case 'Enter': {
          event.preventDefault()
          console.log('[PromptForm] Handling Enter')
          const selectedItem =
            currentSuggestion.items[suggestionState?.selectedIndex ?? 0]
          if (selectedItem) {
            currentSuggestion.command({
              id: selectedItem.id,
              label: selectedItem.label,
              category: selectedItem.category
            })
          }
          return true
        }

        default:
          return false
      }
    },
    [suggestionState?.selectedIndex]
  )

  useEffect(() => {
    function handleCategoryKeyDown(e: KeyboardEvent) {
      if (suggestionState) return

      if (menuState.view === 'categories') {
        console.log('[PromptForm] Key pressed in categories:', e.key)

        switch (e.key) {
          case 'ArrowUp': {
            e.preventDefault()
            setCategorySelectedIndex(
              prev => (prev - 1 + categoryItems.length) % categoryItems.length
            )
            break
          }
          case 'ArrowDown': {
            e.preventDefault()
            setCategorySelectedIndex(prev => (prev + 1) % categoryItems.length)
            break
          }
          case 'Enter': {
            e.preventDefault()
            const selectedCategory = categoryItems[categorySelectedIndex]
            setMenuState({
              view: selectedCategory.category === 'file' ? 'files' : 'symbols',
              category: selectedCategory.category
            })
            break
          }
          default:
            break
        }
      }
    }

    window.addEventListener('keydown', handleCategoryKeyDown)
    return () => window.removeEventListener('keydown', handleCategoryKeyDown)
  }, [suggestionState, menuState, categoryItems, categorySelectedIndex])

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
      MentionExtension.configure({
        HTMLAttributes: {
          class: 'mention'
        },
        suggestion: {
          char: '@',
          allowSpaces: true,
          items: async ({ query }) => {
            console.log('[MentionExtension] Fetching items for query:', query)
            if (!provideFileAtInfo) return []
            try {
              const files = await provideFileAtInfo({ query })
              console.log('[MentionExtension] Files fetched:', files?.length)
              if (!files) return []
              return files.map(file => ({
                type: 'source',
                category: 'file' as const,
                id: file.name,
                label: file.name,
                filepath: file.filepath,
                data: {
                  sourceId: file.name,
                  sourceName: file.name,
                  sourceKind: 'file'
                }
              }))
            } catch (error) {
              console.error('[MentionExtension] Error fetching files:', error)
              return []
            }
          },
          render: () => ({
            onStart: props => {
              console.log('[MentionExtension] Suggestion started')
              const newState = {
                items: props.items,
                command: props.command,
                clientRect: props.clientRect!,
                selectedIndex: 0
              }
              suggestionRef.current = {
                items: props.items,
                command: props.command
              }
              setSuggestionState(newState)
            },
            onUpdate: props => {
              console.log('[MentionExtension] Suggestion updated')
              const newState = {
                items: props.items,
                command: props.command,
                clientRect: props.clientRect!,
                selectedIndex: 0
              }
              suggestionRef.current = {
                items: props.items,
                command: props.command
              }
              setSuggestionState(newState)
            },
            onKeyDown: ({ event }) => {
              console.log(
                '[MentionExtension] Key down in suggestion:',
                event.key
              )
              if (['ArrowUp', 'ArrowDown', 'Enter'].includes(event.key)) {
                return handleKeyDown(event)
              }
              return false
            },
            onExit: () => {
              console.log('[MentionExtension] Exiting suggestion')
              setMenuState({ view: 'categories' })
              suggestionRef.current = null
              setSuggestionState(null)
            }
          })
        }
      })
    ],
    content: input,
    onUpdate: ({ editor }) => {
      console.log('[PromptForm] Editor content updated')
      setInput(editor.getText())
    }
  })

  React.useImperativeHandle(ref, () => ({
    focus: () => {
      console.log('[PromptForm] Focus requested')
      editor?.commands.focus()
    }
  }))

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    console.log('[PromptForm] Form submitted')

    if (!input?.trim() || isLoading || isInitializing) {
      console.log(
        '[PromptForm] Submit prevented - empty input or loading state'
      )
      return
    }

    await onSubmit(input)
    editor?.commands.setContent('')
  }

  return (
    <>
      <form onSubmit={handleSubmit} ref={formRef}>
        <div className="bg-background relative flex max-h-60 w-full grow flex-col overflow-hidden px-8 sm:rounded-md sm:border sm:px-12">
          <Button
            variant="ghost"
            size="icon"
            className="bg-background hover:bg-background absolute left-0 top-4 h-8 w-8 rounded-full p-0 sm:left-4"
          >
            <span className="sr-only">Edit message</span>
          </Button>

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

      {suggestionState && (
        <Popover open={true} modal={false}>
          <PopoverContent
            ref={popoverRef}
            className="p-0 w-[280px] overflow-y-auto"
            style={{
              position: 'absolute',
              left: suggestionState.clientRect()?.left ?? 0,
              top:
                (suggestionState.clientRect()?.top ?? 0) -
                (menuState.view === 'categories'
                  ? 70
                  : menuState.view === 'files' &&
                    suggestionState.items.length > 0
                  ? Math.min(suggestionState.items.length * 42, 4 * 42)
                  : 70),
              height: 'auto',
              maxHeight: '200px'
            }}
            align="start"
            onOpenAutoFocus={e => e.preventDefault()}
            onPointerDownOutside={e => e.preventDefault()}
            onFocusOutside={e => e.preventDefault()}
          >
            {menuState.view === 'categories' ? (
              <CategoryMenu
                items={categoryItems}
                selectedIndex={categorySelectedIndex}
                onSelect={cat => {
                  console.log('[PromptForm] Category selected:', cat)
                  setMenuState({
                    view: cat === 'file' ? 'files' : 'symbols',
                    category: cat
                  })
                }}
                onUpdateSelectedIndex={index => {
                  setCategorySelectedIndex(index)
                }}
              />
            ) : menuState.view === 'files' ? (
              <FileList
                items={suggestionState.items}
                selectedIndex={suggestionState.selectedIndex}
                onSelect={item => {
                  console.log('[PromptForm] File selected:', item)
                  suggestionState.command(item)
                }}
                onUpdateSelectedIndex={updateSelectedIndex}
              />
            ) : (
              <div className="h-full flex items-center justify-center px-3 py-2.5 text-sm text-muted-foreground/70">
                Symbol search coming soon...
              </div>
            )}
          </PopoverContent>
        </Popover>
      )}
    </>
  )
}

export const PromptForm = React.forwardRef<PromptFormRef, PromptProps>(
  PromptFormRenderer
)

export default PromptForm
