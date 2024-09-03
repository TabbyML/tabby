'use client'

import { useEffect, useRef, useState } from 'react'
import { Editor } from '@tiptap/react'

import { ContextInfo } from '@/lib/gql/generates/graphql'
import { useCurrentTheme } from '@/lib/hooks/use-current-theme'
import { ThreadRunContexts } from '@/lib/types'
import {
  cn,
  getMentionsFromText,
  getThreadRunContextsFromMentions
} from '@/lib/utils'
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger
} from '@/components/ui/tooltip'

import { PromptEditor, PromptEditorRef } from './prompt-editor'
import { IconArrowRight, IconSpinner } from './ui/icons'

export default function TextAreaSearch({
  onSearch,
  className,
  placeholder,
  showBetaBadge,
  isLoading,
  autoFocus,
  loadingWithSpinning,
  cleanAfterSearch = true,
  isFollowup,
  contextInfo,
  fetchingContextInfo
}: {
  onSearch: (value: string, ctx: ThreadRunContexts) => void
  className?: string
  placeholder?: string
  showBetaBadge?: boolean
  isLoading?: boolean
  autoFocus?: boolean
  loadingWithSpinning?: boolean
  cleanAfterSearch?: boolean
  isFollowup?: boolean
  contextInfo?: ContextInfo
  fetchingContextInfo: boolean
}) {
  const [isShow, setIsShow] = useState(false)
  const [isFocus, setIsFocus] = useState(false)
  const [value, setValue] = useState('')
  const { theme } = useCurrentTheme()
  const editorRef = useRef<PromptEditorRef>(null)

  useEffect(() => {
    // Ensure the textarea height remains consistent during rendering
    setIsShow(true)
  }, [])

  const onWrapperClick = () => {
    if (isFollowup) {
      editorRef.current?.editor?.commands.focus()
    }
  }

  const handleSubmit = (editor: Editor | undefined | null) => {
    if (!editor) {
      return
    }

    const text = editor.getText()
    const mentions = getMentionsFromText(text, contextInfo?.sources)
    const ctx = getThreadRunContextsFromMentions(mentions)

    // do submit
    onSearch(text, ctx)

    // clear content
    if (cleanAfterSearch) {
      editorRef.current?.editor?.chain().clearContent().focus().run()
    }
  }

  const handleClickMentionIcon = () => {
    const editor = editorRef.current?.editor
    if (editor) {
      const { state } = editor
      const { selection } = state
      const { from } = selection

      // const $from = state.doc.resolve(from)
      // const type = state.schema.nodes['mention']
      // const allow = !!$from.parent.type.contentMatch.matchType(type)
      const charBeforeCursor = state.doc.textBetween(from - 1, from, ' ')
      const isAtLineStart =
        from === 1 || state.doc.textBetween(from - 1, from, '\n') === '\n'
      const hasSpaceBeforeCursor = charBeforeCursor === ' '

      // FIXME
      if (isAtLineStart || hasSpaceBeforeCursor) {
        editor.chain().focus().insertContent('@').run()
      } else {
        editor.chain().focus().insertContent(' @').run()
      }
    }
  }

  const showFooterToolbar = false

  return (
    <div
      className={cn(
        'relative flex w-full items-center overflow-hidden rounded-lg border border-muted-foreground bg-background px-4 transition-all hover:border-muted-foreground/60',
        {
          '!border-zinc-400': isFocus && isFollowup && theme !== 'dark',
          '!border-primary': isFocus && (!isFollowup || theme === 'dark'),
          'py-0': showBetaBadge,
          'border-2 dark:border border-zinc-400 hover:border-zinc-400/60 dark:border-muted-foreground dark:hover:border-muted-foreground/60':
            isFollowup
        },
        className
      )}
      onClick={onWrapperClick}
    >
      {showBetaBadge && (
        <Tooltip delayDuration={0}>
          <TooltipTrigger asChild>
            <span
              className="absolute -right-8 top-1 mr-3 rotate-45 rounded-none border-none py-0.5 pl-6 pr-5 text-xs text-primary"
              style={{ background: theme === 'dark' ? '#333' : '#e8e1d3' }}
            >
              Beta
            </span>
          </TooltipTrigger>
          <TooltipContent sideOffset={-8} className="max-w-md">
            <p>
              Please note that the answer engine is still in its early stages,
              and certain functionalities, such as finding the correct code
              context and the quality of summarizations, still have room for
              improvement. If you encounter an issue and believe it can be
              enhanced, consider sharing it in our Slack community!
            </p>
          </TooltipContent>
        </Tooltip>
      )}
      <PromptEditor
        editable
        contextInfo={contextInfo}
        fetchingContextInfo={fetchingContextInfo}
        onSubmit={handleSubmit}
        placeholder={placeholder || 'Ask anything...'}
        autoFocus={autoFocus}
        onFocus={() => setIsFocus(true)}
        onBlur={() => setIsFocus(false)}
        onUpdate={({ editor }) => setValue(editor.getText())}
        ref={editorRef}
        placement={isFollowup ? 'bottom' : 'top'}
        className={cn(
          'text-area-autosize mr-1 flex-1 resize-none rounded-lg !border-none bg-transparent !shadow-none !outline-none !ring-0 !ring-offset-0',
          {
            '!h-[48px]': !isShow,
            'pt-4': !showBetaBadge,
            'pt-5': showBetaBadge,
            'pb-4': !showFooterToolbar && !showBetaBadge,
            'pb-5': !showFooterToolbar && showBetaBadge
          }
        )}
        editorClassName={isFollowup ? 'min-h-[1.75rem]' : 'min-h-[3.5rem]'}
      />
      <div
        className={cn('flex items-center justify-between gap-2', {
          'pb-2': showFooterToolbar
        })}
      >
        <div
          className={cn(
            'flex items-center justify-center rounded-lg p-1 transition-all',
            {
              'bg-primary text-primary-foreground cursor-pointer':
                value.length > 0,
              '!bg-muted !text-primary !cursor-default':
                isLoading || value.length === 0,
              'mr-1.5': !showBetaBadge,
              'h-6 w-6': !isFollowup
            }
          )}
          onClick={() => handleSubmit(editorRef.current?.editor)}
        >
          {loadingWithSpinning && isLoading && (
            <IconSpinner className="h-3.5 w-3.5" />
          )}
          {(!loadingWithSpinning || !isLoading) && (
            <IconArrowRight className="h-3.5 w-3.5" />
          )}
        </div>
      </div>
    </div>
  )
}
