'use client'

import { useMemo, useRef, useState } from 'react'
import { Editor } from '@tiptap/react'
import { Maybe } from 'graphql/jsutils/Maybe'

import { NEWLINE_CHARACTER } from '@/lib/constants'
import { ContextInfo } from '@/lib/gql/generates/graphql'
import { useCurrentTheme } from '@/lib/hooks/use-current-theme'
import { ThreadRunContexts } from '@/lib/types'
import {
  checkSourcesAvailability,
  cn,
  getMentionsFromText,
  getThreadRunContextsFromMentions,
  isCodeSourceContext
} from '@/lib/utils'
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger
} from '@/components/ui/tooltip'

import LoadingWrapper from '../loading-wrapper'
import { PromptEditor, PromptEditorRef } from '../prompt-editor'
import { Button } from '../ui/button'
import { IconArrowRight, IconAtSign, IconSpinner } from '../ui/icons'
import { Separator } from '../ui/separator'
import { Skeleton } from '../ui/skeleton'
import { ModelSelect } from './model-select'
import { RepoSelect } from './repo-select'

export default function TextAreaSearch({
  onSearch,
  modelName,
  onSelectModel,
  repoSourceId,
  onSelectRepo,
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
  isInitializingResources,
  models
}: {
  onSearch: (value: string, ctx: ThreadRunContexts) => void
  onSelectModel: (v: string) => void
  isInitializingResources: boolean
  // selected model
  modelName: string | undefined
  // selected repo
  repoSourceId: string | undefined
  onSelectRepo: (id: string | undefined) => void
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
    onSelectModel(v)
    setTimeout(() => {
      focusTextarea()
    })
  }

  const handleSelectRepo = (id: string | undefined) => {
    onSelectRepo(id)
    setTimeout(() => {
      focusTextarea()
    })
  }

  const handleSubmit = (editor: Editor | undefined | null) => {
    if (!editor || isLoading || isInitializingResources) {
      return
    }

    const text = editor
      .getText({
        blockSeparator: NEWLINE_CHARACTER
      })
      .trim()
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

  const { hasDocumentSource } = useMemo(() => {
    return checkSourcesAvailability(contextInfo?.sources)
  }, [contextInfo?.sources])

  const repos = useMemo(() => {
    return contextInfo?.sources.filter(x => isCodeSourceContext(x.sourceKind))
  }, [contextInfo?.sources])

  const showModelSelect = !!models?.length
  const showRepoSelect =
    !!repos?.length && (!isFollowup || (isFollowup && !!repoSourceId))
  const showBottomBar = showModelSelect || showRepoSelect

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
        className={cn('flex items-end pr-4', {
          'min-h-[5.5rem]': !isFollowup,
          'min-h-[2.5rem]': isFollowup
        })}
      >
        <div className="mr-1 flex-1 overflow-x-hidden pl-4">
          <PromptEditor
            editable
            contextInfo={contextInfo}
            fetchingContextInfo={fetchingContextInfo}
            onSubmit={handleSubmit}
            placeholder={placeholder || 'Ask anything...'}
            autoFocus={autoFocus}
            onFocus={() => setIsFocus(true)}
            onBlur={() => setIsFocus(false)}
            onUpdate={({ editor }) =>
              setValue(
                editor
                  .getText({
                    blockSeparator: NEWLINE_CHARACTER
                  })
                  .trim()
              )
            }
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
              isFollowup && showBottomBar ? 'min-h-[1.75rem]' : 'min-h-[3.5em]'
            }
          />
          {isFollowup && showBottomBar && (
            <div
              className="-ml-2 mb-2 flex items-center gap-2"
              onClick={e => e.stopPropagation()}
            >
              {showRepoSelect && (
                <RepoSelect
                  isInitializing={fetchingContextInfo}
                  repos={repos}
                  value={repoSourceId}
                  onChange={handleSelectRepo}
                  disabled={isFollowup}
                />
              )}
              {showRepoSelect && showModelSelect && (
                <Separator orientation="vertical" className="h-5" />
              )}
              {showModelSelect && (
                <ModelSelect
                  isInitializing={isInitializingResources}
                  models={models}
                  value={modelName}
                  onChange={handleSelectModel}
                />
              )}
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
                  isLoading || value.length === 0 || isInitializingResources,
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
            loading={isInitializingResources || fetchingContextInfo}
            delay={0}
            fallback={
              <div className="flex h-8 w-[40%] items-center">
                <Skeleton className="h-4 w-full" />
              </div>
            }
          >
            {/* mention docs */}
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  className="gap-2 px-1.5 py-1 text-foreground/90"
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

            {/* select codebase */}
            <Separator orientation="vertical" className="h-5" />
            <RepoSelect
              repos={repos}
              value={repoSourceId}
              onChange={handleSelectRepo}
              isInitializing={fetchingContextInfo}
            />

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
