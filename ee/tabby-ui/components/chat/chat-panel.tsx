import React, { RefObject, useMemo, useState } from 'react'
import slugify from '@sindresorhus/slugify'
import { Content, EditorEvents } from '@tiptap/core'
import { useWindowSize } from '@uidotdev/usehooks'
import { AnimatePresence, motion } from 'framer-motion'
import { compact } from 'lodash-es'
import { toast } from 'sonner'

import { SLUG_TITLE_MAX_LENGTH } from '@/lib/constants'
import { useCopyToClipboard } from '@/lib/hooks/use-copy-to-clipboard'
import { useLatest } from '@/lib/hooks/use-latest'
import {
  updateEnableActiveSelection,
  useChatStore
} from '@/lib/stores/chat-store'
import { useMutation } from '@/lib/tabby/gql'
import { setThreadPersistedMutation } from '@/lib/tabby/query'
import type { Context } from '@/lib/types'
import {
  cn,
  getFileLocationFromContext,
  getTitleFromMessages,
  resolveFileNameForDisplay
} from '@/lib/utils'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  IconCheck,
  IconEye,
  IconEyeOff,
  IconFileText,
  IconRefresh,
  IconRemove,
  IconShare,
  IconStop,
  IconTerminalSquare,
  IconTrash
} from '@/components/ui/icons'
import { ButtonScrollToBottom } from '@/components/button-scroll-to-bottom'
import PromptForm from '@/components/chat/prompt-form'

import { Tooltip, TooltipContent, TooltipTrigger } from '../ui/tooltip'
import { ChatContext } from './chat-context'
import { RepoSelect } from './repo-select'
import { PromptFormRef } from './types'

export interface ChatPanelProps {
  setInput: (v: string) => void
  id?: string
  className?: string
  onSubmit: (content: string) => Promise<any>
  onUpdate: (p: EditorEvents['update']) => void
  reload: () => void
  chatInputRef: RefObject<PromptFormRef>
  input: string
  stop: () => void
}

export interface ChatPanelRef {
  focus: () => void
  setInput: (input: Content) => void
  input: string
}

export const ChatPanel = React.forwardRef<ChatPanelRef, ChatPanelProps>(
  ({ stop, reload, className, onSubmit, onUpdate, chatInputRef }, ref) => {
    const {
      threadId,
      container,
      onClearMessages,
      qaPairs,
      isLoading,
      relevantContext,
      activeSelection,
      onCopyContent,
      selectedRepoId,
      setSelectedRepoId,
      repos,
      initialized,
      setRelevantContext,
      openInEditor
    } = React.useContext(ChatContext)
    const enableActiveSelection = useChatStore(
      state => state.enableActiveSelection
    )

    const [persisting, setPerisiting] = useState(false)
    const { width } = useWindowSize()
    const isExtraSmallScreen = typeof width === 'number' && width < 376

    const slugWithThreadId = useMemo(() => {
      if (!threadId) return ''
      const content = qaPairs[0]?.user.content
      if (!content) return threadId

      const title = getTitleFromMessages([], content, {
        maxLength: SLUG_TITLE_MAX_LENGTH
      })
      const slug = slugify(title)
      const slugWithThreadId = compact([slug, threadId]).join('-')
      return slugWithThreadId
    }, [qaPairs[0]?.user.content, threadId])

    const setThreadPersisted = useMutation(setThreadPersistedMutation, {
      onError(err) {
        toast.error(err.message)
      }
    })

    const { isCopied, copyToClipboard } = useCopyToClipboard({
      timeout: 2000,
      onCopyContent
    })

    const handleShareThread = async () => {
      if (!threadId) return
      if (isCopied || persisting) return

      try {
        setPerisiting(true)
        const result = await setThreadPersisted({ threadId })
        if (!result?.data?.setThreadPersisted) {
          toast.error(result?.error?.message || 'Failed to share')
        } else {
          let url = new URL(window.location.origin)
          url.pathname = `/search/${slugWithThreadId}`

          copyToClipboard(url.toString())
        }
      } catch (e) {
      } finally {
        setPerisiting(false)
      }
    }

    const removeRelevantContext = useLatest((idx: number) => {
      const editor = chatInputRef.current?.editor
      if (!editor) {
        return
      }

      setRelevantContext(prev => prev.filter((item, index) => index !== idx))

      editor.commands.focus()
    })

    const onSelectRepo = (sourceId: string | undefined) => {
      setSelectedRepoId(sourceId)

      setTimeout(() => {
        chatInputRef.current?.focus()
      })
    }

    React.useImperativeHandle(
      ref,
      () => {
        return {
          focus: () => {
            chatInputRef.current?.focus()
          },
          setInput: str => {
            chatInputRef.current?.setInput(str)
          },
          input: chatInputRef.current?.input ?? ''
        }
      },
      [chatInputRef]
    )

    return (
      <div className={className}>
        <ButtonScrollToBottom container={container} />
        <div className="mx-auto max-w-5xl px-[16px]">
          <div
            className={cn(
              'mb-1 flex h-10 items-center justify-center',
              isExtraSmallScreen ? 'gap-3' : 'gap-2'
            )}
          >
            {isLoading ? (
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="outline"
                    onClick={() => stop()}
                    className="gap-2 bg-background"
                  >
                    <IconStop />
                    {!isExtraSmallScreen && 'Stop generating'}
                  </Button>
                </TooltipTrigger>
                <TooltipContent hidden={!isExtraSmallScreen}>
                  Stop generating
                </TooltipContent>
              </Tooltip>
            ) : (
              qaPairs?.length > 0 && (
                <>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button
                        variant="outline"
                        onClick={() => reload()}
                        className="gap-2 bg-background"
                      >
                        <IconRefresh />
                        {!isExtraSmallScreen && 'Regenerate'}
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent hidden={!isExtraSmallScreen}>
                      Regenerate
                    </TooltipContent>
                  </Tooltip>

                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button
                        variant="outline"
                        className="gap-2 bg-background"
                        onClick={handleShareThread}
                      >
                        {isCopied ? <IconCheck /> : <IconShare />}
                        {!isExtraSmallScreen && 'Share'}
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent hidden={!isExtraSmallScreen}>
                      Share
                    </TooltipContent>
                  </Tooltip>
                </>
              )
            )}
            {qaPairs?.length > 0 && (
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="outline"
                    onClick={onClearMessages}
                    className="gap-2 bg-background"
                  >
                    <IconTrash />
                    {!isExtraSmallScreen && 'Clear'}
                  </Button>
                </TooltipTrigger>
                <TooltipContent hidden={!isExtraSmallScreen}>
                  Clear
                </TooltipContent>
              </Tooltip>
            )}
          </div>
          <div
            id="chat-panel-container"
            className="rounded-md border bg-background px-4 pb-1.5 pt-3 outline-none transition-shadow duration-300 focus-within:ring-1 focus-within:!ring-ring hover:ring-1 hover:ring-ring/60 focus-visible:ring-offset-2"
          >
            <div className="flex w-full flex-wrap gap-1.5">
              <RepoSelect
                id="repo-select"
                className="overflow-hidden"
                value={selectedRepoId}
                onChange={onSelectRepo}
                repos={repos}
                isInitializing={!initialized}
              />
              {activeSelection ? (
                <Badge
                  id="active-selection-badge"
                  variant="outline"
                  className={cn(
                    'inline-flex h-7 flex-nowrap items-center gap-1.5 overflow-hidden rounded-md border pr-0 text-sm font-semibold',
                    {
                      'border-dashed !text-muted-foreground italic line-through':
                        !enableActiveSelection
                    }
                  )}
                >
                  {activeSelection.kind === 'file' && (
                    <IconFileText className="shrink-0" />
                  )}
                  {activeSelection.kind === 'terminal' && (
                    <IconTerminalSquare className="shrink-0" />
                  )}
                  <ContextLabel
                    context={activeSelection}
                    className="flex-1 truncate"
                  />
                  <span className="shrink-0 text-muted-foreground">
                    Current file
                  </span>
                  <Button
                    size="icon"
                    variant="ghost"
                    className="h-7 w-7 shrink-0 rounded-l-none hover:bg-muted/50"
                    onClick={e => {
                      updateEnableActiveSelection(!enableActiveSelection)
                    }}
                  >
                    {enableActiveSelection ? <IconEye /> : <IconEyeOff />}
                  </Button>
                </Badge>
              ) : null}
              <AnimatePresence>
                {relevantContext.map((item, idx) => {
                  // `gitUrl + filepath + range` as unique key for file context
                  // `name + processId + selection.length + idx` as unique key for terminal context
                  const key =
                    item.kind === 'file'
                      ? `${item.gitUrl}_${item.filepath}_${item.range?.start}_${item.range?.end}`
                      : `${item.name}_${item.processId}_${item.selection.length}_${idx}`
                  return (
                    <motion.div
                      key={key}
                      initial={{ opacity: 0, scale: 0.9, y: -5 }}
                      animate={{ opacity: 1, scale: 1, y: 0 }}
                      transition={{
                        ease: 'easeInOut',
                        duration: 0.1
                      }}
                      exit={{ opacity: 0, scale: 0.9, y: -5 }}
                      layout
                      className="overflow-hidden"
                    >
                      <Badge
                        variant="outline"
                        className={cn(
                          'inline-flex h-7 w-full cursor-pointer flex-nowrap items-center gap-1 overflow-hidden rounded-md pr-0 text-sm font-semibold'
                        )}
                        onClick={() => {
                          if (item.kind === 'file') {
                            openInEditor(getFileLocationFromContext(item))
                          }
                        }}
                      >
                        {item.kind === 'file' && (
                          <IconFileText className="shrink-0" />
                        )}
                        {item.kind === 'terminal' && (
                          <IconTerminalSquare className="shrink-0" />
                        )}
                        <ContextLabel context={item} />
                        <Button
                          size="icon"
                          variant="ghost"
                          className="h-7 w-7 shrink-0 rounded-l-none hover:bg-muted/50"
                          onClick={e => {
                            e.stopPropagation()
                            removeRelevantContext.current(idx)
                          }}
                        >
                          <IconRemove />
                        </Button>
                      </Badge>
                    </motion.div>
                  )
                })}
              </AnimatePresence>
            </div>
            <PromptForm
              ref={chatInputRef}
              onSubmit={onSubmit}
              onUpdate={onUpdate}
              isLoading={isLoading}
            />
          </div>
        </div>
      </div>
    )
  }
)
ChatPanel.displayName = 'ChatPanel'

function ContextLabel({
  context,
  className
}: {
  context: Context
  className?: string
}) {
  if (context.kind === 'terminal') {
    return (
      <span className={cn('truncate', className)} title={context.selection}>
        <span className="text-muted-foreground">
          {context.name ?? 'Terminal Selection'}
        </span>
      </span>
    )
  }
  const line = context.range
    ? context.range.start === context.range.end
      ? `:${context.range.start}`
      : `:${context.range.start}-${context.range.end}`
    : ''

  return (
    <span className={cn('truncate', className)}>
      {resolveFileNameForDisplay(context.filepath)}
      {!!context.range && <span className="text-muted-foreground">{line}</span>}
    </span>
  )
}
