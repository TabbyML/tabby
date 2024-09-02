import './styles.css'

import React, { forwardRef, useImperativeHandle, useLayoutEffect } from 'react'
import Document from '@tiptap/extension-document'
// import HardBreak from '@tiptap/extension-hard-break'
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

const CustomKeyboardShortcuts = (onSubmit: (editor: Editor) => void) =>
  Extension.create({
    addKeyboardShortcuts() {
      return {
        Enter: ({ editor }) => {
          onSubmit(editor)
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
  enabledMarkdown?: boolean
  placement?: 'top' | 'bottom'
}

export interface PromptEditorRef {
  editor: Editor | null
}

interface MentionContextValue {
  list?: ContextSource[]
  pending: boolean
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
      editorClassName,
      enabledMarkdown,
      placement
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
        Placeholder.configure({
          placeholder: placeholder || 'Ask anything...'
        }),
        CustomKeyboardShortcuts(handleSubmit),
        CustomMention.configure({
          HTMLAttributes: {
            class: 'mention'
          },
          renderText({ node }) {
            return `[[source:${node.attrs.id}]]`
          },
          suggestion: suggestion({
            placement: placement === 'bottom' ? 'top-start' : 'bottom-start'
          })
        })
      ],
      editorProps: {
        attributes: {
          class: cn(
            'max-h-38 prose min-h-[3.5rem] max-w-none font-sans dark:prose-invert focus:outline-none prose-p:my-0',
            editorClassName
          )
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
          pending: !!fetchingContextInfo
        }}
      >
        <div
          className={cn(
            'text-area-autosize max-h-36 overflow-y-auto pr-1',
            className
          )}
        >
          <EditorContent editor={editor} />
        </div>
      </MentionContext.Provider>
    )
  }
)
PromptEditor.displayName = 'PromptEditor'
