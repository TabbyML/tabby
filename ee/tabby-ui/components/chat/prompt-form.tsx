import React, { useContext, useImperativeHandle, useRef } from 'react'
import Document from '@tiptap/extension-document'
import Mention from '@tiptap/extension-mention'
import Paragraph from '@tiptap/extension-paragraph'
import Placeholder from '@tiptap/extension-placeholder'
import Text from '@tiptap/extension-text'
import {
  Editor,
  EditorContent,
  Extension,
  Range,
  ReactRenderer,
  useEditor
} from '@tiptap/react'

import './prompt-form.css'

import { EditorState } from '@tiptap/pm/state'
import { uniqBy } from 'lodash-es'
import tippy, { GetReferenceClientRect, Instance } from 'tippy.js'

import { NEWLINE_CHARACTER } from '@/lib/constants'
import { useLatest } from '@/lib/hooks/use-latest'
import { useSelectedModel } from '@/lib/hooks/use-models'
import { updateSelectedModel } from '@/lib/stores/chat-store'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { IconArrowRight, IconAtSign } from '@/components/ui/icons'

import { ModelSelect } from '../textarea-search/model-select'
import { ChatContext } from './chat-context'
import {
  MentionList,
  MentionListActions,
  MentionListProps,
  PromptFormMentionExtension
} from './form-editor/mention'
import { fileItemToSourceItem, getMention } from './form-editor/utils'
import { EditorMentionData, PromptFormRef, PromptProps } from './types'

/**
 * It provides the main logic for the chat input with mention functionality.
 */
const PromptForm = React.forwardRef<PromptFormRef, PromptProps>(
  ({ onSubmit, isLoading, onUpdate, className, ...props }, ref) => {
    const {
      listFileInWorkspace,
      readFileContent,
      relevantContext,
      setRelevantContext,
      listSymbols,
      getChanges
    } = useContext(ChatContext)

    const { selectedModel, models } = useSelectedModel()
    // mentionData snapshoot
    const prevMentionsRef = useRef<Array<EditorMentionData>>([])
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
            placeholder: listFileInWorkspace
              ? 'Ask anything, @ to mention'
              : 'Ask anything ...'
          }),
          CustomKeyboardShortcuts(handleSubmit),
          PromptFormMentionExtension.configure({
            deleteTriggerWithBackspace: true,
            // Customize how mention suggestions are fetched and rendered
            suggestion: {
              allow: ({
                state,
                range
              }: {
                editor: Editor
                state: EditorState
                range: Range
                isActive?: boolean
              }) => {
                const $from = state.doc.resolve(range.from)
                const type = state.schema.nodes[Mention.name]
                const allow = !!$from.parent.type.contentMatch.matchType(type)

                return !!listFileInWorkspace && allow
              },
              char: '@', // Trigger character for mention
              items: async ({ query }) => {
                if (!listFileInWorkspace) return []
                const files = await listFileInWorkspace({ query })
                const items = [
                  ...(listSymbols
                    ? [
                        listSymbols ?? {
                          id: 'category',
                          name: 'Files',
                          category: 'category'
                        },
                        {
                          id: 'category',
                          name: 'Symbols',
                          category: 'category'
                        }
                      ]
                    : [])
                ]

                if (
                  getChanges &&
                  (!query || 'changes'.includes(query.toLowerCase()))
                ) {
                  items.push({
                    id: 'command',
                    name: 'changes',
                    category: 'command'
                  })
                }
                items.push(...uniqBy(files.map(fileItemToSourceItem), 'id'))

                return items
              },

              render: () => {
                let component: ReactRenderer<
                  MentionListActions,
                  MentionListProps
                >
                let popup: Instance[]

                return {
                  onStart: props => {
                    component = new ReactRenderer(MentionList, {
                      props: {
                        ...props,
                        listFileInWorkspace,
                        listSymbols,
                        getChanges
                      },
                      editor: props.editor
                    })

                    if (!props.clientRect) {
                      return
                    }

                    popup = tippy('body', {
                      getReferenceClientRect:
                        props.clientRect as GetReferenceClientRect,
                      appendTo: () => document.body,
                      content: component.element,
                      showOnCreate: true,
                      interactive: true,
                      trigger: 'manual',
                      placement: 'top-start',
                      animation: 'shift-away',
                      maxWidth: '90%'
                    })
                  },
                  onUpdate: props => {
                    component.updateProps(props)
                  },
                  onExit: () => {
                    popup[0].destroy()
                    component.destroy()
                  },
                  onKeyDown: props => {
                    if (props.event.key === 'Escape') {
                      popup[0].hide()

                      return true
                    }
                    return component.ref?.onKeyDown(props) ?? false
                  }
                }
              }
            }
          })
        ],
        editorProps: {
          attributes: {
            class: cn(
              'prose min-h-[3.5em] font-sans dark:prose-invert focus:outline-none prose-p:my-0'
            )
          }
        },
        onCreate({ editor }) {
          prevMentionsRef.current = getMention(editor)
        },
        onUpdate(props) {
          onUpdate?.(props)
        }
      },
      [listFileInWorkspace, getChanges]
    )

    // Current text from the editor (for checking if the submit button is disabled)
    const input = editor?.getText() || ''

    const onInsertMention = (prefix: string) => {
      if (!editor) return

      editor
        .chain()
        .focus()
        .command(({ tr, state }) => {
          const { $from } = state.selection
          const isAtLineStart = $from.parentOffset === 0
          const isPrecededBySpace =
            $from.nodeBefore?.text?.endsWith(' ') ?? false

          if (isAtLineStart || isPrecededBySpace) {
            tr.insertText(prefix)
          } else {
            tr.insertText(' ' + prefix)
          }

          return true
        })
        .run()
    }

    const handleSelectModel = (v: string) => {
      updateSelectedModel(v)
      setTimeout(() => {
        editor?.chain().focus().run()
      })
    }

    /**
     * Expose methods to the parent component via ref
     */
    useImperativeHandle(
      ref,
      () => ({
        focus: () => editor?.commands.focus(),
        setInput: value => editor?.commands.setContent(value),
        input,
        editor
      }),
      [editor, input]
    )

    return (
      <div className={cn('relative flex flex-col', className)} {...props}>
        {/* Editor */}
        <div className="relative flex items-start gap-1.5">
          <div
            className="max-h-32 flex-1 overflow-y-auto py-3"
            onClick={() => {
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
        </div>
        <div className="flex items-center justify-between">
          <div className="-ml-1.5 flex items-center gap-2">
            {!!listFileInWorkspace && (
              <Button
                variant="ghost"
                className="h-auto shrink-0 gap-2 p-1.5 text-foreground/90"
                onClick={e => onInsertMention('@')}
              >
                <IconAtSign />
              </Button>
            )}
            <ModelSelect
              models={models}
              value={selectedModel}
              onChange={handleSelectModel}
              triggerClassName="gap-1 py-1 h-auto"
            />
          </div>
          {/* Submit Button */}
          <Button
            className="h-6 w-6"
            size="icon"
            disabled={isLoading || input === ''}
            onClick={handleSubmit}
          >
            <IconArrowRight className="h-3 w-3" />
          </Button>
        </div>
      </div>
    )
  }
)
PromptForm.displayName = 'PromptForm'

/**
 * For convenience, also export it as default
 */
export default PromptForm

function CustomKeyboardShortcuts(onSubmit: () => void) {
  return Extension.create({
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
}
