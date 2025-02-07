'use client'

import { useContext, useState } from 'react'
import { zodResolver } from '@hookform/resolvers/zod'
import DOMPurify from 'dompurify'
import he from 'he'
import { marked } from 'marked'
import { useForm } from 'react-hook-form'
import Textarea from 'react-textarea-autosize'
import * as z from 'zod'

import { MessageAttachmentCode } from '@/lib/gql/generates/graphql'
import { makeFormErrorHandler } from '@/lib/tabby/gql'
import {
  AttachmentCodeItem,
  AttachmentDocItem,
  ExtendedCombinedError,
  FileContext
} from '@/lib/types'
import { cn, formatLineHashForCodeBrowser, getContent } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormMessage
} from '@/components/ui/form'
import {
  IconCheckCircled,
  IconCircleDot,
  IconEdit,
  IconGitMerge,
  IconGitPullRequest,
  IconMore,
  IconRefresh,
  IconSpinner
} from '@/components/ui/icons'
import {
  Sheet,
  SheetClose,
  SheetContent,
  SheetFooter,
  SheetHeader,
  SheetTitle,
  SheetTrigger
} from '@/components/ui/sheet'
import { Skeleton } from '@/components/ui/skeleton'
import { CopyButton } from '@/components/copy-button'
import LoadingWrapper from '@/components/loading-wrapper'
import { MessageMarkdown } from '@/components/message-markdown'
import { SiteFavicon } from '@/components/site-favicon'
import { ListSkeleton } from '@/components/skeleton'
import { UserAvatar } from '@/components/user-avatar'

import { SectionItem } from '../types'
import { PageContext } from './page-context'

export function SectionContent({
  className,
  message,
  isGenerating
}: {
  className?: string
  message: SectionItem
  isGenerating?: boolean
}) {
  const { mode, isPageOwner, pendingSectionIds } = useContext(PageContext)

  const [isEditing, setIsEditing] = useState(false)
  const isPending = pendingSectionIds.has(message.id) && !message.content
  // FIXME
  const sources: any[] = []
  const sourceLen = 0

  const onCodeContextClick = (ctx: FileContext) => {
    if (!ctx.filepath) return
    const url = new URL(`${window.location.origin}/files`)
    const searchParams = new URLSearchParams()
    searchParams.append('redirect_filepath', ctx.filepath)
    searchParams.append('redirect_git_url', ctx.git_url)
    url.search = searchParams.toString()

    const lineHash = ctx.range
      ? formatLineHashForCodeBrowser({
          start: ctx.range.start,
          end: ctx.range.end
        })
      : ''
    if (lineHash) {
      url.hash = lineHash
    }

    window.open(url.toString())
  }

  const onCodeCitationMouseEnter = (index: number) => {}

  const onCodeCitationMouseLeave = (index: number) => {}

  const openCodeBrowserTab = (code: MessageAttachmentCode) => {}

  const onCodeCitationClick = (code: MessageAttachmentCode) => {
    if (code.gitUrl) {
      openCodeBrowserTab(code)
    }
  }

  const onDeleteMessage = (id: string) => {}
  const onUpdateMessage = async (content: string) => {
    return undefined
  }

  const handleUpdateAssistantMessage = async (content: string) => {
    const error = await onUpdateMessage(content)
    if (error) {
      return error
    } else {
      setIsEditing(false)
    }
  }

  return (
    <div className={cn('flex flex-col gap-y-5', className)}>
      <LoadingWrapper loading={isPending} fallback={<ListSkeleton />}>
        <div>
          {isGenerating && !message.content && (
            <Skeleton className="mt-1 h-40 w-full" />
          )}
          {isEditing ? (
            <MessageContentForm
              message={message.content}
              onCancel={() => setIsEditing(false)}
              onSubmit={handleUpdateAssistantMessage}
            />
          ) : (
            <>
              <MessageMarkdown
                message={message.content}
                onCodeCitationClick={onCodeCitationClick}
                onCodeCitationMouseEnter={onCodeCitationMouseEnter}
                onCodeCitationMouseLeave={onCodeCitationMouseLeave}
                canWrapLongLines={!isGenerating}
                supportsOnApplyInEditorV2={false}
                className="prose-p:my-0.5 prose-ol:my-1 prose-ul:my-1"
              />
              {/* if isEditing, do not display error message block */}
              {/* {message.error && <ErrorMessageBlock error={message.error} />} */}

              {!isGenerating && !isEditing && (
                <div className="mt-3 flex items-center gap-3 text-sm">
                  {sourceLen > 0 && (
                    <Sheet>
                      <SheetTrigger asChild>
                        <div className="cursor-pointer rounded-full border px-2 py-1">
                          {sourceLen} sources
                        </div>
                      </SheetTrigger>
                      <SheetContent className="flex w-[50vw] min-w-[300px] flex-col">
                        <SheetHeader className="border-b">
                          <SheetTitle>Sources</SheetTitle>
                          <SheetClose />
                        </SheetHeader>
                        <div className="flex-1 space-y-3 overflow-y-auto">
                          {sources.map((x, index) => {
                            // FIXME id
                            return <SourceCard source={x} key={index} />
                          })}
                        </div>
                        <SheetFooter>
                          <Button>Remove sources</Button>
                        </SheetFooter>
                      </SheetContent>
                    </Sheet>
                  )}
                  <div className="flex items-center gap-x-3">
                    {mode === 'view' && (
                      <CopyButton
                        className="-ml-1.5 gap-x-1 px-1 font-normal text-muted-foreground"
                        value={message.content}
                        text="Copy"
                      />
                    )}
                    {isPageOwner && mode === 'edit' && (
                      <>
                        <Button
                          className="flex items-center gap-x-1 px-1 font-normal text-muted-foreground"
                          variant="ghost"
                          onClick={e => setIsEditing(true)}
                        >
                          <IconEdit />
                          <p>Edit</p>
                        </Button>
                        <Button
                          className="flex items-center gap-x-1 px-1 font-normal text-muted-foreground"
                          variant="ghost"
                        >
                          <IconRefresh />
                          <p>Regenerate</p>
                        </Button>
                        <DropdownMenu modal={false}>
                          <DropdownMenuTrigger asChild>
                            <Button
                              className="flex items-center gap-x-1 px-1 font-normal text-muted-foreground"
                              variant="ghost"
                            >
                              <IconMore />
                              <p>More</p>
                            </Button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent align="start">
                            <DropdownMenuItem
                              onSelect={() => {
                                onDeleteMessage(message.id)
                              }}
                            >
                              Delete Section
                            </DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      </>
                    )}
                  </div>
                </div>
              )}
            </>
          )}
        </div>
        {isGenerating && (
          <div>
            <IconSpinner />
          </div>
        )}
      </LoadingWrapper>
    </div>
  )
}

function SourceCard({
  source
}: {
  source: AttachmentDocItem | AttachmentCodeItem
}) {
  const { mode } = useContext(PageContext)
  const isEditMode = mode === 'edit'

  const isDoc =
    source.__typename === 'MessageAttachmentIssueDoc' ||
    source.__typename === 'MessageAttachmentPullDoc' ||
    source.__typename === 'MessageAttachmentWebDoc'

  if (isDoc) {
    return (
      <div className="flex items-start gap-2">
        {isEditMode && <Checkbox className="mt-2" />}
        <div
          className="relative flex cursor-pointer flex-col justify-between rounded-lg border bg-card p-3 text-card-foreground hover:bg-card/60"
          onClick={() => window.open(source.link)}
        >
          <DocSourceCard source={source} />
        </div>
      </div>
    )
  }

  return (
    <div className="flex w-full items-start gap-2">
      {isEditMode && <Checkbox className="mt-2" />}
      <div className="relative flex flex-1 cursor-pointer flex-col justify-between rounded-lg border bg-card p-3 text-card-foreground hover:bg-card/60">
        <div className="flex flex-1 flex-col justify-between gap-y-1">
          <div className="flex flex-col gap-y-0.5">
            <p className="line-clamp-1 w-full overflow-hidden text-ellipsis break-all text-xs font-semibold">
              {source.filepath}
            </p>
          </div>
          <div className="flex items-center text-xs text-muted-foreground">
            <div className="flex w-full flex-1 items-center justify-between gap-1">
              <div className="flex items-center">
                <SiteFavicon hostname={source.gitUrl} />
                <p className="ml-1 truncate">{source.gitUrl}</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

function DocSourceCard({ source }: { source: AttachmentDocItem }) {
  const { hostname } = new URL(source.link)
  const isIssue = source.__typename === 'MessageAttachmentIssueDoc'
  const isPR = source.__typename === 'MessageAttachmentPullDoc'
  const author =
    source.__typename === 'MessageAttachmentWebDoc' ? undefined : source.author

  const showAvatar = (isIssue || isPR) && !!author

  return (
    <div className="flex flex-1 flex-col justify-between gap-y-1">
      <div className="flex flex-col gap-y-0.5">
        <p className="line-clamp-1 w-full overflow-hidden text-ellipsis break-all text-xs font-semibold">
          {source.title}
        </p>

        {showAvatar && (
          <div className="flex items-center gap-1 overflow-x-hidden">
            <UserAvatar user={author} className="h-3.5 w-3.5 shrink-0" />
            <p className="truncate text-xs font-medium text-muted-foreground">
              {author?.name}
            </p>
          </div>
        )}
        {!showAvatar && (
          <p
            className={cn(
              ' w-full overflow-hidden text-ellipsis break-all text-xs text-muted-foreground',
              !showAvatar ? 'line-clamp-2' : 'line-clamp-1'
            )}
          >
            {normalizedText(getContent(source))}
          </p>
        )}
      </div>
      <div className="flex items-center text-xs text-muted-foreground">
        <div className="flex w-full flex-1 items-center justify-between gap-1">
          <div className="flex items-center">
            <SiteFavicon hostname={hostname} />
            <p className="ml-1 truncate">
              {hostname.replace('www.', '').split('/')[0]}
            </p>
          </div>
          <div className="flex shrink-0 items-center gap-1">
            {isIssue && (
              <>
                {source.closed ? (
                  <IconCheckCircled className="h-3.5 w-3.5" />
                ) : (
                  <IconCircleDot className="h-3.5 w-3.5" />
                )}
                <span>{source.closed ? 'Closed' : 'Open'}</span>
              </>
            )}
            {isPR && (
              <>
                {source.merged ? (
                  <IconGitMerge className="h-3.5 w-3.5" />
                ) : (
                  <IconGitPullRequest className="h-3.5 w-3.5" />
                )}
                {source.merged ? 'Merged' : 'Open'}
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

function MessageContentForm({
  message,
  onCancel,
  onSubmit
}: {
  message: string
  onCancel: () => void
  onSubmit: (newMessage: string) => Promise<ExtendedCombinedError | void>
}) {
  const formSchema = z.object({
    content: z.string().trim()
  })
  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: { content: message }
  })
  const { isSubmitting } = form.formState
  const [draftMessage] = useState<string | undefined>(message)

  const handleSubmit = async (values: z.infer<typeof formSchema>) => {
    const error = await onSubmit(values.content)

    if (error) {
      makeFormErrorHandler(form)(error)
    }
  }

  return (
    <Form {...form}>
      <form onSubmit={form.handleSubmit(handleSubmit)}>
        <FormField
          control={form.control}
          name="content"
          render={({ field }) => (
            <FormItem>
              <FormControl>
                <Textarea
                  autoFocus
                  minRows={2}
                  maxRows={20}
                  className="w-full rounded-lg border bg-background p-4 outline-ring"
                  {...field}
                />
              </FormControl>
              <FormMessage />
            </FormItem>
          )}
        />
        <div className="my-4 flex items-center justify-between gap-2 px-2">
          <div>
            <FormMessage />
          </div>
          <div className="flex items-center gap-2">
            <Button
              type="button"
              variant="outline"
              onClick={onCancel}
              className="min-w-[2rem]"
            >
              Cancel
            </Button>
            <Button type="submit" disabled={isSubmitting}>
              {isSubmitting && (
                <IconSpinner className="mr-2 h-4 w-4 animate-spin" />
              )}
              Save
            </Button>
          </div>
        </div>
      </form>
    </Form>
  )
}

// Remove HTML and Markdown format
const normalizedText = (input: string) => {
  const sanitizedHtml = DOMPurify.sanitize(input, {
    ALLOWED_TAGS: [],
    ALLOWED_ATTR: []
  })
  const parsed = marked.parse(sanitizedHtml) as string
  const decoded = he.decode(parsed)
  const plainText = decoded.replace(/<\/?[^>]+(>|$)/g, '')
  return plainText
}
