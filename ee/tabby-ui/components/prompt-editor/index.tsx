import './styles.css'

import React, {
  forwardRef,
  useEffect,
  useImperativeHandle,
  useLayoutEffect,
  useMemo,
  useState
} from 'react'
import Document from '@tiptap/extension-document'
import Paragraph from '@tiptap/extension-paragraph'
import Placeholder from '@tiptap/extension-placeholder'
import Text from '@tiptap/extension-text'
import { PluginKey } from '@tiptap/pm/state'
import {
  Editor,
  EditorContent,
  EditorEvents,
  Extension,
  useEditor
} from '@tiptap/react'

import { ContextInfo, ContextSource } from '@/lib/gql/generates/graphql'
import { useLatest } from '@/lib/hooks/use-latest'
import { cn, isCodeSourceContext, isDocSourceContext } from '@/lib/utils'

import { MentionExtension } from './mention-extension'
import suggestion from './suggestion'

const DocumentMentionPluginKey = new PluginKey('mention-doc')
const CodeMentionPluginKey = new PluginKey('mention-code')

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
      placement
    },
    ref
  ) => {
    const [initialized, setInitialized] = useState(!fetchingContextInfo)
    const doSubmit = useLatest((editor: Editor) => {
      if (submitting) return

      const text = editor.getText()
      if (!text) return

      onSubmit?.(editor)
    })

    const handleSubmit = (editor: Editor) => {
      doSubmit.current(editor)
    }

    const hasCodebaseSources = useMemo(() => {
      if (!contextInfo?.sources) {
        return false
      }

      return contextInfo.sources.some(o => isCodeSourceContext(o.sourceKind))
    }, [contextInfo?.sources])

    const hasDocSources = useMemo(() => {
      if (!contextInfo?.sources) {
        return false
      }

      return contextInfo.sources.some(o => isDocSourceContext(o.sourceKind))
    }, [contextInfo?.sources])

    const editor = useEditor(
      {
        editable: !initialized ? false : editable,
        immediatelyRender: false,
        extensions: [
          Document,
          Paragraph,
          Text,
          Placeholder.configure({
            showOnlyWhenEditable: false,
            placeholder: !initialized
              ? 'Loading...'
              : placeholder || 'Ask anything...'
          }),
          CustomKeyboardShortcuts(handleSubmit),
          // for document mention
          MentionExtension.configure({
            deleteTriggerWithBackspace: true,
            HTMLAttributes: {
              class: 'mention'
            },
            suggestion: suggestion({
              category: 'doc',
              char: '@',
              pluginKey: DocumentMentionPluginKey,
              placement: placement === 'bottom' ? 'top-start' : 'bottom-start',
              disabled: !hasDocSources
            })
          }),
          // for codebase mention
          MentionExtension.configure({
            deleteTriggerWithBackspace: true,
            HTMLAttributes: {
              class: 'mention-code'
            },
            suggestion: suggestion({
              category: 'code',
              char: '#',
              pluginKey: CodeMentionPluginKey,
              placement: placement === 'bottom' ? 'top-start' : 'bottom-start',
              disabled: !hasCodebaseSources
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
        onBlur(props) {
          onBlur?.(props)
        },
        onFocus(props) {
          onFocus?.(props)
        },
        onUpdate(props) {
          onUpdate?.(props)
        }
      },
      [initialized]
    )

    useImperativeHandle(ref, () => ({
      editor
    }))

    useLayoutEffect(() => {
      if (editor && autoFocus) {
        editor.commands.focus()
      }
    }, [editor])

    useEffect(() => {
      if (!fetchingContextInfo && !initialized) {
        setInitialized(true)
      }
    }, [fetchingContextInfo])

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
