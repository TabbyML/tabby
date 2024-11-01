'use client'

import {
  Dispatch,
  SetStateAction,
  useEffect,
  useMemo,
  useRef,
  useState
} from 'react'
import { Editor } from '@tiptap/react'

import { ContextInfo } from '@/lib/gql/generates/graphql'
import { useCurrentTheme } from '@/lib/hooks/use-current-theme'
import { useModel } from '@/lib/hooks/use-models'
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
  DropdownMenuIndicator,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
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
  fetchingContextInfo
}: {
  onSearch: (value: string, ctx: ThreadRunContexts) => void
  onModelSelect: Dispatch<SetStateAction<string>>
  modelName: string
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
}) {
  const [isShow, setIsShow] = useState(false)
  const [isFocus, setIsFocus] = useState(false)
  const [value, setValue] = useState('')
  const { theme } = useCurrentTheme()
  const editorRef = useRef<PromptEditorRef>(null)
  const { data: modelInfo } = useModel()
  const isSelectModelEnabled = true

  const DropdownMenuItems = modelInfo?.chat.map(model => (
    <DropdownMenuRadioItem
      onClick={() => onModelSelect(model)}
      value={model}
      key={model}
      className="cursor-pointer py-2 pl-3"
    >
      <DropdownMenuIndicator className="DropdownMenuItemIndicator">
        <IconArrowRight />
      </DropdownMenuIndicator>
      <span className="ml-2">{model}</span>
    </DropdownMenuRadioItem>
  ))

  useEffect(() => {
    // Ensure the textarea height remains consistent during rendering
    setIsShow(true)
  }, [])

  useEffect(() => {
    if (!modelName) {
      onModelSelect(modelInfo?.chat?.length ? modelInfo?.chat[0] : '')
    }
  }, [modelInfo])

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
    const ctx: ThreadRunContexts = {
      ...getThreadRunContextsFromMentions(mentions),
      modelName
    }

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
      <div
        className={cn('flex items-end px-4', {
          'min-h-[5.5rem]': !isFollowup,
          'min-h-[2.5rem]': isFollowup
        })}
      >
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
            'text-area-autosize mr-1 flex-1 resize-none rounded-lg !border-none bg-transparent !shadow-none !outline-none !ring-0 !ring-offset-0',
            {
              '!h-[48px]': !isShow,
              'py-3': !showBetaBadge,
              'py-4': showBetaBadge
            }
          )}
          // editorClassName={isFollowup ? 'min-h-[3.45rem]' : 'min-h-[3.5em]'}
          editorClassName="min-h-[3.5em]"
        />
        <div className={cn('flex items-center justify-between gap-2')}>
          <div
            className={cn(
              'mb-3 flex items-center justify-center rounded-lg p-1 transition-all',
              {
                'bg-primary text-primary-foreground cursor-pointer':
                  value.length > 0,
                '!bg-muted !text-primary !cursor-default':
                  isLoading || value.length === 0,
                'mr-1.5': !showBetaBadge
                // 'mb-4': !showBetaBadge,
                // 'mb-5': showBetaBadge
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
      <div
        className={cn(
          'hidden items-center gap-2 border-t bg-[#F9F6EF] py-2 pl-2 pr-4 dark:border-muted-foreground/60 dark:bg-[#333333]',
          {
            flex: !isFollowup
          }
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

        {isSelectModelEnabled && modelInfo?.chat?.length && (
          <Separator orientation="vertical" className="h-5" />
        )}

        {/* llm select */}
        <DropdownMenu>
          <DropdownMenuTrigger>
            {isSelectModelEnabled && modelInfo?.chat?.length && (
              <Button
                variant="ghost"
                className="gap-2 px-1.5 py-1 text-foreground/70"
              >
                <IconBox />
                {modelName}
              </Button>
            )}
          </DropdownMenuTrigger>
          <DropdownMenuContent
            side="bottom"
            align="end"
            className="dropdown-menu max-h-[30vh] min-w-[20rem] overflow-y-auto overflow-x-hidden rounded-md border bg-popover p-2 text-popover-foreground shadow animate-in"
          >
            <DropdownMenuRadioGroup
              value={modelName}
              onValueChange={onModelSelect}
            >
              {DropdownMenuItems}
            </DropdownMenuRadioGroup>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </div>
  )
}
