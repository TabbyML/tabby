'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import { Editor } from '@tiptap/react'

import { ContextInfo } from '@/lib/gql/generates/graphql'
import { useCurrentTheme } from '@/lib/hooks/use-current-theme'
import { ThreadRunContexts } from '@/lib/types'
import {
  checkSourcesAvailability,
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
import { Button } from './ui/button'
import {
  IconArrowRight,
  IconAtSign,
  IconBox,
  IconHash,
  IconSpinner
} from './ui/icons'
import { Separator } from './ui/separator'

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
    editorRef.current?.editor?.commands.focus()
  }

  const handleSubmit = (editor: Editor | undefined | null) => {
    if (!editor || isLoading) {
      return
    }

    const text = editor.getText().trim()
    if (!text) return

    const mentions = getMentionsFromText(text, contextInfo?.sources)
    const ctx = getThreadRunContextsFromMentions(mentions)

    // do submit
    onSearch(text, ctx)

    // clear content
    if (cleanAfterSearch) {
      editorRef.current?.editor?.chain().clearContent().focus().run()
    }
  }

  const onInsertMention = (prefix: string) => {
    const editor = editorRef.current?.editor
    if (!editor) return

    editor.chain().focus().insertContent(prefix).run()
  }

  const { hasCodebaseSource, hasDocumentSource } = useMemo(() => {
    return checkSourcesAvailability(contextInfo?.sources)
  }, [contextInfo?.sources])

  return (
    <div
      className={cn(
        'relative overflow-hidden border bg-background transition-all rounded-xl hover:border-primary/80',
        {
          'border-primary/80': isFocus
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
      <div className="flex items-center px-4 min-h-[6rem]">
        <PromptEditor
          editable
          contextInfo={contextInfo}
          fetchingContextInfo={fetchingContextInfo}
          onSubmit={handleSubmit}
          // placeholder={
          //   placeholder ||
          //   (contextInfo?.sources?.length
          //     ? 'Ask anything...\n\nUse # to select a codebase to chat with, or @ to select a document to bring into context.'
          //     : 'Ask anything...')
          // }
          placeholder={placeholder || 'Ask anything...'}
          autoFocus={autoFocus}
          onFocus={() => setIsFocus(true)}
          onBlur={() => setIsFocus(false)}
          onUpdate={({ editor }) => setValue(editor.getText().trim())}
          ref={editorRef}
          placement={isFollowup ? 'bottom' : 'top'}
          className={cn(
            'text-area-autosize mr-1 flex-1 resize-none rounded-lg !border-none bg-transparent !shadow-none !outline-none !ring-0 !ring-offset-0',
            {
              '!h-[48px]': !isShow,
              'py-4': !showBetaBadge,
              'py-5': showBetaBadge
            }
          )}
          editorClassName={isFollowup ? 'min-h-[1.725rem]' : 'min-h-[3.5em]'}
        />
        <div className={cn('flex items-center justify-between gap-2')}>
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
      <div
        className={cn(
          'bg-popover/50 pr-4 pl-2 py-2 border-t flex items-center gap-2'
        )}
        onClick={e => e.stopPropagation()}
      >
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="ghost"
              className="gap-2 px-1.5 py-1 text-foreground/70"
              onClick={e => onInsertMention('#')}
              disabled={!hasCodebaseSource}
            >
              <IconHash />
              Codebase
            </Button>
          </TooltipTrigger>
          <TooltipContent className="max-w-md">
            Select a codebase to chat with
          </TooltipContent>
        </Tooltip>

        <Separator orientation="vertical" className="h-5" />
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="ghost"
              className="gap-2 px-1.5 py-1 text-foreground/70"
              onClick={e => onInsertMention('@')}
              disabled={!hasDocumentSource}
            >
              <IconAtSign />
              Documents
            </Button>
          </TooltipTrigger>
          <TooltipContent className="max-w-md">
            Select a document to bring into context
          </TooltipContent>
        </Tooltip>
      </div>
    </div>
  )
}
