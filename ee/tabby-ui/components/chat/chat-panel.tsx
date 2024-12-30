import React, { RefObject, useMemo, useState } from 'react'
import slugify from '@sindresorhus/slugify'
import { useWindowSize } from '@uidotdev/usehooks'
import type { UseChatHelpers } from 'ai/react'
import { AnimatePresence, motion } from 'framer-motion'
import { compact } from 'lodash-es'
import { toast } from 'sonner'

import { SLUG_TITLE_MAX_LENGTH } from '@/lib/constants'
import { useCopyToClipboard } from '@/lib/hooks/use-copy-to-clipboard'
import { updateEnableActiveSelection } from '@/lib/stores/chat-actions'
import { useChatStore } from '@/lib/stores/chat-store'
import { useMutation } from '@/lib/tabby/gql'
import { setThreadPersistedMutation } from '@/lib/tabby/query'
import type { Context } from '@/lib/types'
import {
  cn,
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
  IconTrash
} from '@/components/ui/icons'
import { ButtonScrollToBottom } from '@/components/button-scroll-to-bottom'
import { PromptForm, PromptFormRef } from '@/components/chat/prompt-form'
import { FooterText } from '@/components/footer'

import { Tooltip, TooltipContent, TooltipTrigger } from '../ui/tooltip'
import { ChatContext } from './chat'
import { RepoSelect } from './repo-select'

export interface ChatPanelProps
  extends Pick<UseChatHelpers, 'stop' | 'input' | 'setInput'> {
  id?: string
  className?: string
  onSubmit: (content: string) => Promise<any>
  reload: () => void
  chatMaxWidthClass: string
  chatInputRef: RefObject<HTMLTextAreaElement>
}

export interface ChatPanelRef {
  focus: () => void
}

function ChatPanelRenderer(
  {
    stop,
    reload,
    input,
    setInput,
    className,
    onSubmit,
    chatMaxWidthClass,
    chatInputRef
  }: ChatPanelProps,
  ref: React.Ref<ChatPanelRef>
) {
  const promptFormRef = React.useRef<PromptFormRef>(null)
  const {
    threadId,
    container,
    onClearMessages,
    qaPairs,
    isLoading,
    relevantContext,
    removeRelevantContext,
    activeSelection,
    onCopyContent,
    selectedRepoId,
    setSelectedRepoId,
    repos,
    initialized
  } = React.useContext(ChatContext)
  const enableActiveSelection = useChatStore(
    state => state.enableActiveSelection
  )

  const [persisting, setPerisiting] = useState(false)
  const { width } = useWindowSize()
  const isExtraSmallScreen = typeof width === 'number' && width < 376

  const slugWithThreadId = useMemo(() => {
    if (!threadId) return ''
    const content = qaPairs[0]?.user.message
    if (!content) return threadId

    const title = getTitleFromMessages([], content, {
      maxLength: SLUG_TITLE_MAX_LENGTH
    })
    const slug = slugify(title)
    const slugWithThreadId = compact([slug, threadId]).join('-')
    return slugWithThreadId
  }, [qaPairs[0]?.user.message, threadId])

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
          promptFormRef.current?.focus()
        }
      }
    },
    []
  )

  return (
    <div className={className}>
      <ButtonScrollToBottom container={container} />
      <div className={`mx-auto md:px-4 ${chatMaxWidthClass}`}>
        <div
          className={cn(
            'flex h-10 items-center justify-center',
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
        <div className="border-t bg-background px-4 py-2 shadow-lg sm:space-y-4 sm:rounded-t-xl sm:border md:py-4">
          <div className="flex flex-wrap gap-2">
            <AnimatePresence presenceAffectsLayout>
              <RepoSelect
                value={selectedRepoId}
                onChange={onSelectRepo}
                repos={repos}
                isInitializing={!initialized}
              />
              {activeSelection ? (
                <motion.div
                  key="active-selection"
                  initial={{ opacity: 0, scale: 0.9, y: -5 }}
                  animate={{ opacity: 1, scale: 1, y: 0 }}
                  transition={{
                    ease: 'easeInOut',
                    duration: 0.1
                  }}
                  exit={{ opacity: 0, scale: 0.9, y: 5 }}
                >
                  <Badge
                    variant="outline"
                    className={cn(
                      'inline-flex h-7 flex-nowrap items-center gap-1.5 overflow-hidden rounded-md pr-0 text-sm font-semibold',
                      {
                        'border-dashed !text-muted-foreground italic line-through':
                          !enableActiveSelection
                      }
                    )}
                  >
                    <IconFileText />
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
                      className="h-7 w-7 shrink-0 rounded-l-none"
                      onClick={e => {
                        updateEnableActiveSelection(!enableActiveSelection)
                      }}
                    >
                      {enableActiveSelection ? <IconEye /> : <IconEyeOff />}
                    </Button>
                  </Badge>
                </motion.div>
              ) : null}
              {relevantContext.map((item, idx) => {
                // `git_url + filepath + range` as unique key
                const key = `${item.git_url}_${item.filepath}_${item.range?.start}_${item.range?.end}`
                return (
                  <motion.div
                    key={key}
                    initial={{ opacity: 0, scale: 0.9, y: -5 }}
                    animate={{ opacity: 1, scale: 1, y: 0 }}
                    transition={{
                      ease: 'easeInOut',
                      duration: 0.1
                    }}
                    exit={{ opacity: 0, scale: 0.9, y: 5 }}
                    layout
                  >
                    <Badge
                      variant="outline"
                      className="inline-flex h-7 flex-nowrap items-center gap-1 overflow-hidden rounded-md pr-0 text-sm font-semibold"
                    >
                      <ContextLabel context={item} />
                      <Button
                        size="icon"
                        variant="ghost"
                        className="h-7 w-7 shrink-0 rounded-l-none"
                        onClick={removeRelevantContext.bind(null, idx)}
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
            ref={promptFormRef}
            onSubmit={onSubmit}
            input={input}
            setInput={setInput}
            isLoading={isLoading}
            chatInputRef={chatInputRef}
            isInitializing={!initialized}
          />
          <FooterText className="hidden sm:block" />
        </div>
      </div>
    </div>
  )
}

export const ChatPanel = React.forwardRef<ChatPanelRef, ChatPanelProps>(
  ChatPanelRenderer
)

function ContextLabel({
  context,
  className
}: {
  context: Context
  className?: string
}) {
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
