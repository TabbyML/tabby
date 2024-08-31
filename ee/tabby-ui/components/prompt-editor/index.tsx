import './styles.css'

import React, { forwardRef, useImperativeHandle, useLayoutEffect } from 'react'
import Document from '@tiptap/extension-document'
import HardBreak from '@tiptap/extension-hard-break'
import Mention from '@tiptap/extension-mention'
import Paragraph from '@tiptap/extension-paragraph'
import Placeholder from '@tiptap/extension-placeholder'
import Text from '@tiptap/extension-text'
import {
  Editor,
  EditorContent,
  EditorEvents,
  Extension,
  useEditor
} from '@tiptap/react'

import { ContextInfo, ContextSource } from '@/lib/gql/generates/graphql'
import { useLatest } from '@/lib/hooks/use-latest'
import { cn } from '@/lib/utils'

import { CustomMention } from './custom-mention-extension'
import suggestion from './suggestion'

const DisableEnter = (onSubmit: (editor: Editor) => void) =>
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
  contextInfo?: ContextInfo
  fetchingContextInfo?: boolean
  submitting?: boolean
  onSubmit?: (editor: Editor) => void
  placeholder?: string
  onBlur?: (p: EditorEvents['blur']) => void
  onFocus?: (p: EditorEvents['focus']) => void
  onUpdate?: (p: EditorEvents['update']) => void
  autoFocus?: boolean
  className?: string
  editorClassName?: string
}

export interface PromptEditorRef {
  editor: Editor | null
}

interface MentionContextValue {
  list?: ContextSource[]
  pending: boolean
  canSearchPublic: boolean
}

export const MentionContext = React.createContext<MentionContextValue>(
  {} as MentionContextValue
)

export const PromptEditor = forwardRef<PromptEditorRef, PromptEditorProps>(
  (
    {
      editable,
      content,
      contextInfo,
      fetchingContextInfo,
      submitting,
      onSubmit,
      placeholder,
      onBlur,
      onFocus,
      onUpdate,
      autoFocus,
      className,
      editorClassName
    },
    ref
  ) => {
    const doSubmit = useLatest((editor: Editor) => {
      if (submitting) return

      const text = editor.getText()
      if (!text) return

      onSubmit?.(editor)
    })

    const handleSubmit = (editor: Editor) => {
      doSubmit.current(editor)
    }

    const editor = useEditor({
      extensions: [
        Document,
        Paragraph,
        Text,
        HardBreak,
        Placeholder.configure({
          placeholder: placeholder || 'Ask anything...'
        }),
        DisableEnter(handleSubmit),
        CustomMention.configure({
          HTMLAttributes: {
            class: 'mention'
          },
          suggestion
        })
      ],
      editorProps: {
        attributes: {
          class: cn(
            'prose dark:prose-invert prose-p:my-0 focus:outline-none font-sans max-w-none max-h-38 min-h-[3.5rem]',
            editorClassName
          )
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
      editable,
      onBlur(props) {
        onBlur?.(props)
      },
      onFocus(props) {
        onFocus?.(props)
      },
      onUpdate(props) {
        onUpdate?.(props)
      }
    })

    useImperativeHandle(ref, () => ({
      editor
    }))

    useLayoutEffect(() => {
      if (editor && autoFocus) {
        editor.commands.focus()
      }
    }, [editor])

    if (!editor) {
      return null
    }

    return (
      <MentionContext.Provider
        value={{
          list: contextInfo?.sources,
          // FIXME
          canSearchPublic: !!contextInfo?.canSearchPublic,
          pending: !!fetchingContextInfo
        }}
      >
        <div
          className={cn(
            'text-area-autosize pr-1 max-h-36 overflow-y-auto',
            className
          )}
        >
          <EditorContent editor={editor} />
        </div>
      </MentionContext.Provider>
    )
  }
)
