'use client'

import { useEffect, useState } from 'react'
import TextareaAutosize from 'react-textarea-autosize'

import { cn } from '@/lib/utils'

import { IconArrowRight } from './ui/icons'

export default function TextAreaSearch({
  onSearch,
  className
}: {
  onSearch: (value: string) => void
  className?: string
}) {
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
    if (!value) return
    onSearch(value)
    setValue('')
  }

  return (
    <div
      className={cn(
        'flex w-full items-center rounded-lg border border-muted-foreground bg-background transition-all hover:border-muted-foreground/60',
        {
          '!border-primary': isFocus
        },
        className
      )}
    >
      <TextareaAutosize
        className={cn(
          'flex-1 resize-none rounded-lg !border-none bg-transparent px-4 py-3 !shadow-none !outline-none !ring-0 !ring-offset-0',
          {
            '!h-[48px]': !isShow
          }
        )}
        placeholder="Ask anything"
        maxRows={5}
        onKeyDown={onSearchKeyDown}
        onKeyUp={onSearchKeyUp}
        onFocus={() => setIsFocus(true)}
        onBlur={() => setIsFocus(false)}
        onChange={e => setValue(e.target.value)}
        value={value}
      />
      <div
        className={cn(
          'mr-3 flex items-center rounded-lg bg-muted p-1 text-muted-foreground transition-all',
          {
            '!bg-primary !text-primary-foreground': value.length > 0
          }
        )}
        onClick={search}
      >
        <IconArrowRight className="h-3.5 w-3.5" />
      </div>
    </div>
  )
}
