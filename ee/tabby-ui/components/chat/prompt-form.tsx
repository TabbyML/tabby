import React, {
  ForwardedRef,
  useContext,
  useEffect,
  useImperativeHandle
} from 'react'
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
import { isEqual, uniqBy } from 'lodash-es'
import { EditorFileContext } from 'tabby-chat-panel/index'
import tippy, { GetReferenceClientRect, Instance } from 'tippy.js'

import { NEWLINE_CHARACTER } from '@/lib/constants'
import { useDebounceCallback } from '@/lib/hooks/use-debounce'
import { useLatest } from '@/lib/hooks/use-latest'
import { FileContext } from '@/lib/types'
import { cn, convertEditorContext } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { IconArrowElbow, IconEdit } from '@/components/ui/icons'

import { ChatContext } from './chat'
import { emitter } from './event-emitter'
import {
  MentionList,
  MentionListActions,
  MentionListProps,
  PromptFormMentionExtension
} from './form-editor/mention'
import { PromptFormRef, PromptProps } from './form-editor/types'
import { fileItemToSourceItem, isSameFileContext } from './form-editor/utils'

/**
 * PromptFormRenderer is the internal component used by React.forwardRef
 * It provides the main logic for the chat input with mention functionality.
 */
function PromptFormRenderer(
  { onSubmit, isLoading, onUpdate }: PromptProps,
  ref: ForwardedRef<PromptFormRef>
) {
  const {
    listFileInWorkspace,
    readFileContent,
    relevantContext,
    setRelevantContext,
    listSymbols
  } = useContext(ChatContext)

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
                  : []),
                ...uniqBy(files.map(fileItemToSourceItem), 'id')
              ]
              return items
            },

            render: () => {
              let component: ReactRenderer<MentionListActions, MentionListProps>
              let popup: Instance[]

              return {
                onStart: props => {
                  component = new ReactRenderer(MentionList, {
                    props: { ...props, listFileInWorkspace, listSymbols },
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
                    animation: 'shift-away'
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
            'prose max-w-none font-sans dark:prose-invert focus:outline-none prose-p:my-0'
          )
        }
      },
      onUpdate(props) {
        onUpdate?.(props)
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
      input,
      editor
    }),
    [editor, input]
  )

  /**
   * This function compares the current mentions in the editor with the relevant context
   * and updates the context accordingly.
   * It adds new mentions and removes mentions that are no longer present in the editor.
   * Only mentions that refer to the whole file are considered.
   */
  const diffAndUpdateMentionContext = useDebounceCallback(async () => {
    if (!readFileContent || !editor) return

    const contextInEditor: EditorFileContext[] = []
    editor.view.state.doc.descendants(node => {
      if (
        node.type.name === 'mention' &&
        (node.attrs.category === 'file' || node.attrs.category === 'symbol')
      ) {
        contextInEditor.push({
          kind: 'file',
          content: '',
          filepath: node.attrs.fileItem.filepath,
          range:
            node.attrs.category === 'symbol'
              ? node.attrs.fileItem.range
              : undefined
        })
      }
    })

    let prevContext: FileContext[] = relevantContext
    let updatedContext = [...prevContext]

    const mentionsToAdd = contextInEditor.filter(
      ctx =>
        !prevContext.some(prevCtx =>
          isSameFileContext(convertEditorContext(ctx), prevCtx)
        )
    )

    // Remove mentions from the context if they are no longer present in the editor
    const mentionsToRemove = prevContext.filter(
      prevCtx =>
        !contextInEditor.some(ctx =>
          isSameFileContext(convertEditorContext(ctx), prevCtx)
        )
    )

    for (const ctx of mentionsToRemove) {
      updatedContext = updatedContext.filter(prevCtx => !isEqual(prevCtx, ctx))
    }

    for (const ctx of mentionsToAdd) {
      // Read the file content and add it to the context
      const content = await readFileContent({
        filepath: ctx.filepath,
        range: ctx.range
      })
      updatedContext.push(
        convertEditorContext({
          kind: 'file',
          content: content || '',
          filepath: ctx.filepath,
          range: ctx.range
        })
      )
    }

    setRelevantContext(updatedContext)
  }, 100)

  useEffect(() => {
    const onFileMentionUpdate = () => {
      diffAndUpdateMentionContext.run()
    }

    emitter.on('file_mention_update', onFileMentionUpdate)

    return () => {
      emitter.off('file_mention_update', onFileMentionUpdate)
    }
  }, [])

  return (
    <div className="relative flex flex-col px-2.5">
      {/* Editor & Submit row */}
      <div className="relative flex items-start gap-2">
        <span className="mt-[1.375rem]">
          <IconEdit className="h-4 w-4" />
        </span>
        <div
          className="max-h-32 flex-1 overflow-y-auto py-4"
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
