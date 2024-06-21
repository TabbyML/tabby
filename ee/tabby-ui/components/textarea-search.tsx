'use client'

import { useEffect, useState } from 'react'
import { useTheme } from 'next-themes'
import TextareaAutosize from 'react-textarea-autosize'

import { cn } from '@/lib/utils'

import { IconArrowRight } from './ui/icons'

export default function TextAreaSearch({
  onSearch,
  className,
  placeholder,
  showBetaBadge,
  isLoading,
  autoFocus
}: {
  onSearch: (value: string) => void
  className?: string
  placeholder?: string
  showBetaBadge?: boolean
  isLoading?: boolean
  autoFocus?: boolean
}) {
  const [isShow, setIsShow] = useState(false)
  const [isFocus, setIsFocus] = useState(false)
  const [value, setValue] = useState('')
  const { theme } = useTheme()

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
        'relative flex w-full items-center overflow-hidden rounded-lg border border-muted-foreground bg-background p-4 transition-all hover:border-muted-foreground/60',
        {
          '!border-primary': isFocus,
          'py-5': showBetaBadge
        },
        className
      )}
    >
      {showBetaBadge && (
        <span
          className="absolute -right-8 top-1 mr-3 rotate-45 rounded-none border-none py-0.5 pl-6 pr-5 text-xs text-primary"
          style={{ background: theme === 'dark' ? '#333' : '#e8e1d3' }}
        >
          Beta
        </span>
      )}
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
        autoFocus={autoFocus}
      />
      <div
        className={cn('mr-6 flex items-center rounded-lg p-1 transition-all', {
          'bg-primary text-primary-foreground cursor-pointer': value.length > 0,
          '!bg-muted !text-primary !cursor-default':
            isLoading || value.length === 0,
          'mr-6': showBetaBadge,
          'mr-1.5': !showBetaBadge
        })}
        onClick={search}
      >
        <IconArrowRight className="h-3.5 w-3.5" />
      </div>
    </div>
  )
}
