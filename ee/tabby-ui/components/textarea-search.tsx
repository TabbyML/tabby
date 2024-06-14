'use client'

import { useEffect, useState } from 'react'
import TextareaAutosize from 'react-textarea-autosize'

import { useHealth } from '@/lib/hooks/use-health'
import { cn } from '@/lib/utils'

import { IconArrowRight, IconBox } from './ui/icons'

export default function TextAreaSearch({
  onSearch,
  className,
  placeholder,
  isExpanded,
  isLoading
}: {
  onSearch: (value: string) => void
  className?: string
  placeholder?: string
  isExpanded?: boolean
  isLoading?: boolean
}) {
  const { data: healthInfo } = useHealth()
  const [isShow, setIsShow] = useState(false)
  const [isFocus, setIsFocus] = useState(false)
  const [value, setValue] = useState('')

  useEffect(() => {
    // Ensure the textarea height remains consistent during rendering
    setIsShow(true)
  }, [])

  const onSearchKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) return e.preventDefault()
  }

  const onSearchKeyUp = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      return search()
    }
  }

  const search = () => {
    if (!value || isLoading) return
    onSearch(value)
    setValue('')
  }

  return (
    <div
      className={cn(
        'flex w-full rounded-lg border border-muted-foreground bg-background p-4 transition-all hover:border-muted-foreground/60',
        {
          '!border-primary': isFocus,
          'items-center': !isExpanded,
          'flex-col': isExpanded
        },
        className
      )}
    >
      <TextareaAutosize
        className={cn(
          'text-area-autosize flex-1 resize-none rounded-lg !border-none bg-transparent !shadow-none !outline-none !ring-0 !ring-offset-0',
          {
            '!h-[48px]': !isShow
          }
        )}
        placeholder={placeholder || 'Ask anything...'}
        maxRows={5}
        onKeyDown={onSearchKeyDown}
        onKeyUp={onSearchKeyUp}
        onFocus={() => setIsFocus(true)}
        onBlur={() => setIsFocus(false)}
        onChange={e => setValue(e.target.value)}
        value={value}
      />
      {!isExpanded && (
        <div
          className={cn(
            'mr-3 flex items-center rounded-lg p-1 transition-all',
            {
              'bg-primary text-primary-foreground cursor-pointer':
                value.length > 0,
              '!bg-muted !text-muted-foreground !cursor-default':
                isLoading || value.length === 0
            }
          )}
          onClick={search}
        >
          <IconArrowRight className="h-3.5 w-3.5" />
        </div>
      )}
      {isExpanded && (
        <div className="mt-3 flex items-center text-xs">
          <div className="flex-1">
            <div className="flex items-center text-muted-foreground">
              <IconBox className="mr-1 h-3.5 w-3.5" />
              <p>{healthInfo!.chat_model}</p>
            </div>
          </div>
          <div
            className={cn(
              'mr-3 flex items-center rounded-lg p-1 transition-all',
              {
                'text-primary cursor-pointer': value.length > 0,
                '!text-muted-foreground !cursor-default':
                  isLoading || value.length === 0
              }
            )}
            onClick={search}
          >
            <IconArrowRight className="h-3.5 w-3.5" />
          </div>
        </div>
      )}
    </div>
  )
}
