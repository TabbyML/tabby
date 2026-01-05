import * as React from 'react'
import { X } from 'lucide-react'

import { cn } from '@/lib/utils'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { InputProps } from '@/components/ui/input'

interface TagInputProps extends Omit<InputProps, 'value' | 'onChange'> {
  value?: string[]
  onChange?: (value: string[]) => void
}

const TagInput = React.forwardRef<HTMLInputElement, TagInputProps>(
  ({ className, value = [], onChange, ...props }, ref) => {
    const [inputValue, setInputValue] = React.useState('')

    const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
      if (e.key === 'Enter') {
        e.preventDefault()
        const newTag = inputValue.trim()
        if (newTag && !value.includes(newTag)) {
          onChange?.([...value, newTag])
          setInputValue('')
        }
      } else if (e.key === 'Backspace' && !inputValue && value.length > 0) {
        onChange?.(value.slice(0, -1))
      }
    }

    const removeTag = (tagToRemove: string) => {
      onChange?.(value.filter(tag => tag !== tagToRemove))
    }

    return (
      <div
        className={cn(
          'flex min-h-[2.25rem] w-full flex-wrap items-center gap-2 rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm ring-offset-background focus-within:ring-2 focus-within:ring-ring focus-within:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50',
          className
        )}
      >
        {value.map(tag => (
          <Badge key={tag} variant="secondary" className="gap-1 pr-1">
            {tag}
            <Button
              type="button"
              variant="ghost"
              size="icon"
              className="h-3 w-3 hover:bg-transparent"
              onClick={() => removeTag(tag)}
            >
              <X className="h-3 w-3" />
              <span className="sr-only">Remove {tag}</span>
            </Button>
          </Badge>
        ))}
        <input
          ref={ref}
          type="text"
          className="flex-1 bg-transparent outline-none placeholder:text-muted-foreground"
          value={inputValue}
          onChange={e => setInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
          {...props}
        />
      </div>
    )
  }
)
TagInput.displayName = 'TagInput'

export { TagInput }
