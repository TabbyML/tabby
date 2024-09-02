'use client'

import { useEffect, useRef, useState } from 'react'
import { Editor } from '@tiptap/react'

import { ContextInfo, ContextKind } from '@/lib/gql/generates/graphql'
import { useCurrentTheme } from '@/lib/hooks/use-current-theme'
import { AnswerEngineExtraContext } from '@/lib/types'
import { cn, isCodeSourceContext } from '@/lib/utils'
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger
} from '@/components/ui/tooltip'

import { PromptEditor, PromptEditorRef } from './prompt-editor'
import { getMentionsWithIndices } from './prompt-editor/utils'
import { buttonVariants } from './ui/button'
import { IconArrowRight, IconAtSign, IconSpinner } from './ui/icons'

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
  onSearch: (value: string, ctx: AnswerEngineExtraContext) => void
  className?: string
  placeholder?: string
  showBetaBadge?: boolean
  isLoading?: boolean
  autoFocus?: boolean
  loadingWithSpinning?: boolean
  cleanAfterSearch?: boolean
  isFollowup?: boolean
  extraContext?: AnswerEngineExtraContext
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
    const mentions = getMentionsWithIndices(editor)
    const docSourceIds: string[] = []
    const codeSourceIds: string[] = []
    let searchPublic = false
    for (let mention of mentions) {
      const { kind, id } = mention
      if (isCodeSourceContext(kind)) {
        codeSourceIds.push(id)
      } else if (kind === ContextKind.Web) {
        searchPublic = true
      } else {
        docSourceIds.push(id)
      }
    }

    const ctx: AnswerEngineExtraContext = {
      searchPublic,
      docSourceIds,
      codeSourceIds
    }

    // do submit
    onSearch(text, ctx)

    // clear content
    if (cleanAfterSearch) {
      editorRef.current?.editor?.chain().clearContent().focus().run()
    }
  }

  const showFooterToolbar = true

  return (
    <div
      className={cn(
        'relative overflow-hidden rounded-lg border border-muted-foreground bg-background px-4 transition-all hover:border-muted-foreground/60',
        {
          'flex-col gap-1 w-full': showFooterToolbar,
          'flex w-full items-center ': !showFooterToolbar,
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
        <div className="flex items-center gap-4">
          <div
            className={cn(
              buttonVariants({ variant: 'ghost' }),
              '-ml-2 cursor-pointer rounded-full px-2',
              className
            )}
            onClick={() => {
              editorRef.current?.editor
                ?.chain()
                .insertContent(' @')
                .focus()
                .run()
            }}
          >
            <div className="flex items-center gap-1 overflow-hidden">
              <IconAtSign className={cn('shrink-0 text-foreground/60')} />
              <span className={cn('flex-1 truncate text-foreground/60')}>
                Add Context
              </span>
            </div>
          </div>
        </div>
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
              // 'mr-6': showBetaBadge,
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
