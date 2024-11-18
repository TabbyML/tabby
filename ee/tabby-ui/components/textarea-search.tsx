'use client'

import { useMemo, useRef, useState } from 'react'
import { Editor } from '@tiptap/react'
import { Maybe } from 'graphql/jsutils/Maybe'

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
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger
} from '@/components/ui/tooltip'

import LoadingWrapper from './loading-wrapper'
import { PromptEditor, PromptEditorRef } from './prompt-editor'
import { Button } from './ui/button'
import {
  IconArrowRight,
  IconAtSign,
  IconBox,
  IconCheck,
  IconHash,
  IconSpinner
} from './ui/icons'
import { Separator } from './ui/separator'
import { Skeleton } from './ui/skeleton'

export default function TextAreaSearch({
  onSearch,
  onModelSelect,
  modelName,
  className,
  placeholder,
  showBetaBadge,
  isLoading,
  autoFocus,
  loadingWithSpinning,
  cleanAfterSearch = true,
  isFollowup,
  contextInfo,
  fetchingContextInfo,
  isModelLoading,
  models
}: {
  onSearch: (value: string, ctx: ThreadRunContexts) => void
  onModelSelect: (v: string) => void
  isModelLoading: boolean
  modelName: string | undefined
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
  onValueChange?: (value: string | undefined) => void
  models: Maybe<Array<string>> | undefined
}) {
  const [isFocus, setIsFocus] = useState(false)
  const [value, setValue] = useState('')
  const editorRef = useRef<PromptEditorRef>(null)

  const focusTextarea = () => {
    editorRef.current?.editor?.commands.focus()
  }

  const onWrapperClick = () => {
    focusTextarea()
  }

  const handleSelectModel = (v: string) => {
    onModelSelect(v)
    setTimeout(() => {
      focusTextarea()
    })
  }

  const handleSubmit = (editor: Editor | undefined | null) => {
    if (!editor || isLoading || isModelLoading) {
      return
    }

    const text = editor.getText().trim()
    if (!text) return

    const mentions = getMentionsFromText(text, contextInfo?.sources)
    const ctx: ThreadRunContexts = {
      ...getThreadRunContextsFromMentions(mentions),
      modelName
    }

    // do submit
    onSearch(text, ctx)

    // clear content
    if (cleanAfterSearch) {
      editorRef.current?.editor?.chain().clearContent().focus().run()
      setValue('')
    }
  }

  const onInsertMention = (prefix: string) => {
    const editor = editorRef.current?.editor
    if (!editor) return

    editor
      .chain()
      .focus()
      .command(({ tr, state }) => {
        const { $from } = state.selection
        const isAtLineStart = $from.parentOffset === 0
        const isPrecededBySpace = $from.nodeBefore?.text?.endsWith(' ') ?? false

        if (isAtLineStart || isPrecededBySpace) {
          tr.insertText(prefix)
        } else {
          tr.insertText(' ' + prefix)
        }

        return true
      })
      .run()
  }

  const { hasCodebaseSource, hasDocumentSource } = useMemo(() => {
    return checkSourcesAvailability(contextInfo?.sources)
  }, [contextInfo?.sources])

  const showModelSelect = !!models?.length

  return (
    <div
      className={cn(
        'relative w-full overflow-hidden rounded-xl border bg-background transition-all hover:border-ring dark:border-muted-foreground/60 dark:hover:border-muted-foreground',
        {
          'border-ring dark:border-muted-foreground': isFocus
        },
        className
      )}
      onClick={onWrapperClick}
    >
      {showBetaBadge && <BetaBadge />}

      <div
        className={cn('flex items-end px-4', {
          'min-h-[5.5rem]': !isFollowup,
          'min-h-[2.5rem]': isFollowup
        })}
      >
        <div className="mr-1 flex-1 overflow-x-hidden">
          <PromptEditor
            editable
            contextInfo={contextInfo}
            fetchingContextInfo={fetchingContextInfo}
            onSubmit={handleSubmit}
            placeholder={placeholder || 'Ask anything...'}
            autoFocus={autoFocus}
            onFocus={() => setIsFocus(true)}
            onBlur={() => setIsFocus(false)}
            onUpdate={({ editor }) => setValue(editor.getText().trim())}
            ref={editorRef}
            placement={isFollowup ? 'bottom' : 'top'}
            className={cn(
              'text-area-autosize resize-none rounded-lg !border-none bg-transparent !shadow-none !outline-none !ring-0 !ring-offset-0',
              {
                'py-3': !showBetaBadge,
                'py-4': showBetaBadge
              }
            )}
            editorClassName={
              isFollowup && showModelSelect
                ? 'min-h-[1.75rem]'
                : 'min-h-[3.5em]'
            }
          />
          {isFollowup && showModelSelect && (
            <div className="-ml-2 mb-2 flex">
              <ModelSelect
                isInitializing={isModelLoading}
                models={models}
                value={modelName}
                onChange={handleSelectModel}
              />
            </div>
          )}
        </div>
        <div className={cn('mb-3 flex items-center justify-between gap-2')}>
          <div
            className={cn(
              'flex items-center justify-center rounded-lg p-1 transition-all',
              {
                'bg-primary text-primary-foreground cursor-pointer':
                  value.length > 0,
                '!bg-muted !text-primary !cursor-default':
                  isLoading || value.length === 0 || isModelLoading,
                'mr-1.5': !showBetaBadge
              }
            )}
            onClick={() => handleSubmit(editorRef.current?.editor)}
          >
            {loadingWithSpinning && isLoading && (
              <IconSpinner className="h-4 w-4" />
            )}
            {(!loadingWithSpinning || !isLoading) && (
              <IconArrowRight className="h-4 w-4" />
            )}
          </div>
        </div>
      </div>

      {/* bottom toolbar for HomePage */}
      {!isFollowup && (
        <div
          className={cn(
            'flex items-center gap-2 border-t bg-[#F9F6EF] py-2 pl-2 pr-4 dark:border-muted-foreground/60 dark:bg-[#333333]'
          )}
          onClick={e => e.stopPropagation()}
        >
          <LoadingWrapper
            loading={isModelLoading || fetchingContextInfo}
            delay={0}
            fallback={
              <div className="flex h-8 w-[40%] items-center">
                <Skeleton className="h-4 w-full" />
              </div>
            }
          >
            {/* mention codebase */}
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

            {/* mention docs */}
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

            {/* model select */}
            {!!models?.length && (
              <>
                <Separator orientation="vertical" className="h-5" />
                <ModelSelect
                  models={models}
                  value={modelName}
                  onChange={handleSelectModel}
                />
              </>
            )}
          </LoadingWrapper>
        </div>
      )}
    </div>
  )
}

interface ModelSelectProps {
  models: Maybe<Array<string>> | undefined
  value: string | undefined
  onChange: (v: string) => void
  isInitializing?: boolean
}

function ModelSelect({
  models,
  value,
  onChange,
  isInitializing
}: ModelSelectProps) {
  const onModelSelect = (v: string) => {
    onChange(v)
  }

  return (
    <LoadingWrapper
      loading={isInitializing}
      fallback={
        <div className="w-full pl-2">
          <Skeleton className="h-3 w-[20%]" />
        </div>
      }
    >
      {!!models?.length && (
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button
              variant="ghost"
              className="gap-2 px-1.5 py-1 text-foreground/70"
            >
              <IconBox />
              {value}
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent
            side="bottom"
            align="start"
            className="dropdown-menu max-h-[30vh] min-w-[20rem] overflow-y-auto overflow-x-hidden rounded-md border bg-popover p-2 text-popover-foreground shadow animate-in"
          >
            <DropdownMenuRadioGroup value={value} onValueChange={onChange}>
              {models.map(model => {
                const isSelected = model === value
                return (
                  <DropdownMenuRadioItem
                    onClick={e => {
                      onModelSelect(model)
                      e.stopPropagation()
                    }}
                    value={model}
                    key={model}
                    className="cursor-pointer py-2 pl-3"
                  >
                    <IconCheck
                      className={cn(
                        'mr-2 shrink-0',
                        model === value ? 'opacity-100' : 'opacity-0'
                      )}
                    />
                    <span
                      className={cn({
                        'font-medium': isSelected
                      })}
                    >
                      {model}
                    </span>
                  </DropdownMenuRadioItem>
                )
              })}
            </DropdownMenuRadioGroup>
          </DropdownMenuContent>
        </DropdownMenu>
      )}
    </LoadingWrapper>
  )
}

function BetaBadge() {
  const { theme } = useCurrentTheme()
  return (
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
          Please note that the answer engine is still in its early stages, and
          certain functionalities, such as finding the correct code context and
          the quality of summarizations, still have room for improvement. If you
          encounter an issue and believe it can be enhanced, consider sharing it
          in our Slack community!
        </p>
      </TooltipContent>
    </Tooltip>
  )
}
