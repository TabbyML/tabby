'use client'

import { useRef, useState } from 'react'
import { Editor } from '@tiptap/react'

import { NEWLINE_CHARACTER } from '@/lib/constants'
import { useCurrentTheme } from '@/lib/hooks/use-current-theme'
import { cn } from '@/lib/utils'
import { IconArrowRight, IconSpinner } from '@/components/ui/icons'
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger
} from '@/components/ui/tooltip'
import { PromptEditor, PromptEditorRef } from '@/components/prompt-editor'

export default function SectionForm({
  onSearch,
  className,
  placeholder,
  showBetaBadge,
  isLoading,
  autoFocus,
  loadingWithSpinning,
  cleanAfterSearch = true
}: {
  onSearch: (value: string) => void
  className?: string
  placeholder?: string
  showBetaBadge?: boolean
  isLoading?: boolean
  autoFocus?: boolean
  loadingWithSpinning?: boolean
  cleanAfterSearch?: boolean
  onValueChange?: (value: string | undefined) => void
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

  const handleSubmit = (editor: Editor | undefined | null) => {
    if (!editor || isLoading) {
      return
    }

    const text = editor
      .getText({
        blockSeparator: NEWLINE_CHARACTER
      })
      .trim()
    if (!text) return

    // do submit
    onSearch(text)

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

      <div className={cn('flex min-h-[2.5rem] items-end pr-4')}>
        <div className="mr-1 flex-1 overflow-x-hidden pl-4">
          <PromptEditor
            editable
            contextInfo={undefined}
            fetchingContextInfo={false}
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
            placement={'bottom'}
            className={cn(
              'text-area-autosize resize-none rounded-lg !border-none bg-transparent !shadow-none !outline-none !ring-0 !ring-offset-0',
              {
                'py-3': !showBetaBadge,
                'py-4': showBetaBadge
              }
            )}
          />
        </div>
        <div className={cn('mb-3 flex items-center justify-between gap-2')}>
          <div
            className={cn(
              'flex items-center justify-center rounded-lg p-1 transition-all',
              {
                'bg-primary text-primary-foreground cursor-pointer':
                  value.length > 0,
                '!bg-muted !text-primary !cursor-default':
                  isLoading || value.length === 0,
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
