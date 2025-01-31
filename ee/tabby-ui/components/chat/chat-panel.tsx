import React, { RefObject, useMemo, useState } from 'react'
import slugify from '@sindresorhus/slugify'
import { Content, EditorEvents } from '@tiptap/core'
import { useWindowSize } from '@uidotdev/usehooks'
import type { UseChatHelpers } from 'ai/react'
import { AnimatePresence, motion } from 'framer-motion'
import { compact } from 'lodash-es'
import { toast } from 'sonner'

import { SLUG_TITLE_MAX_LENGTH } from '@/lib/constants'
import { useCopyToClipboard } from '@/lib/hooks/use-copy-to-clipboard'
import { useLatest } from '@/lib/hooks/use-latest'
import { updateEnableActiveSelection } from '@/lib/stores/chat-actions'
import { useChatStore } from '@/lib/stores/chat-store'
import { useMutation } from '@/lib/tabby/gql'
import { setThreadPersistedMutation } from '@/lib/tabby/query'
import type { Context, FileContext } from '@/lib/types'
import {
  cn,
  convertEditorContext,
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
  IconFile,
  IconFileText,
  IconRefresh,
  IconRemove,
  IconShare,
  IconStop,
  IconTrash
} from '@/components/ui/icons'
import { ButtonScrollToBottom } from '@/components/button-scroll-to-bottom'
import { PromptForm } from '@/components/chat/prompt-form'
import { FooterText } from '@/components/footer'

import { Tooltip, TooltipContent, TooltipTrigger } from '../ui/tooltip'
import { ChatContext } from './chat'
import { PromptFormRef } from './form-editor/types'
import { isSameFileContext } from './form-editor/utils'
import { RepoSelect } from './repo-select'

export interface ChatPanelProps extends Pick<UseChatHelpers, 'stop' | 'input'> {
  setInput: (v: string) => void
  id?: string
  className?: string
  onSubmit: (content: string) => Promise<any>
  onUpdate: (p: EditorEvents['update']) => void
  reload: () => void
  chatMaxWidthClass: string
  chatInputRef: RefObject<PromptFormRef>
}

export interface ChatPanelRef {
  focus: () => void
  setInput: (input: Content) => void
  input: string
}

function ChatPanelRenderer(
  {
    stop,
    reload,
    className,
    onSubmit,
    onUpdate,
    chatMaxWidthClass,
    chatInputRef
  }: ChatPanelProps,
  ref: React.Ref<ChatPanelRef>
) {
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

  const removeRelevantContext = useLatest((idx: number) => {
    const editor = chatInputRef.current?.editor
    if (!editor) {
      return
    }

    const { state, view } = editor
    const { tr } = state
    const positionsToDelete: any[] = []

    const currentContext: FileContext = relevantContext[idx]
    state.doc.descendants((node, pos) => {
      // TODO: use a easy way to dealling with mention node
      if (
        node.type.name === 'mention' &&
        (node.attrs.category === 'file' || node.attrs.category === 'symbol')
      ) {
        const fileContext = convertEditorContext({
          filepath: node.attrs.fileItem.filepath,
          content: '',
          kind: 'file',
          range:
            node.attrs.category === 'symbol'
              ? node.attrs.fileItem.range
              : undefined
        })
        if (isSameFileContext(fileContext, currentContext)) {
          positionsToDelete.push({ from: pos, to: pos + node.nodeSize })
        }
      }
    })

    setRelevantContext(prev => prev.filter((item, index) => index !== idx))
    positionsToDelete.reverse().forEach(({ from, to }) => {
      tr.delete(from, to)
    })

    view.dispatch(tr)
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
        <div
          id="chat-panel-container"
          className="border-t bg-background px-4 py-2 shadow-lg sm:space-y-4 sm:rounded-t-xl sm:border md:py-4"
        >
          <div className="flex flex-wrap gap-2">
            <RepoSelect
              id="repo-select"
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
                    exit={{ opacity: 0, scale: 0.9, y: -5 }}
                    layout
                  >
                    <Badge
                      variant="outline"
                      className={cn(
                        'inline-flex h-7 cursor-pointer flex-nowrap items-center gap-1 overflow-hidden rounded-md pr-0 text-sm font-semibold'
                      )}
                      onClick={() => {
                        openInEditor(getFileLocationFromContext(item))
                      }}
                    >
                      <IconFile className="shrink-0" />
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
