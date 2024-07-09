import React from 'react'
import { useRouter, useSearchParams } from 'next/navigation'

import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { IconClose, IconSearch } from '@/components/ui/icons'
import { Input } from '@/components/ui/input'

import { SourceCodeBrowserContext } from './source-code-browser'
import { generateEntryPath } from './utils'

interface CodeSearchBarProps extends React.OlHTMLAttributes<HTMLDivElement> {}

export const CodeSearchBar: React.FC<CodeSearchBarProps> = ({ className }) => {
  const router = useRouter()
  const searchParams = useSearchParams()
  const { activeEntryInfo, activeRepo } = React.useContext(
    SourceCodeBrowserContext
  )
  const [query, setQuery] = React.useState(searchParams.get('q')?.toString())
  const inputRef = React.useRef<HTMLInputElement>(null)
  const clearInput = () => {
    setQuery('')
    inputRef.current?.focus()
  }

  // shortcut 's'
  React.useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      const target = event.target as Element
      const tagName = target?.tagName?.toLowerCase()
      if (
        tagName === 'input' ||
        tagName === 'textarea' ||
        tagName === 'select'
      ) {
        return
      }

      if (event.key === 's') {
        event.preventDefault()
        inputRef.current?.focus()
      }
    }

    window.addEventListener('keydown', handleKeyDown)

    return () => {
      window.removeEventListener('keydown', handleKeyDown)
    }
  }, [])

  const onSubmit: React.FormEventHandler<HTMLFormElement> = e => {
    e.preventDefault()

    if (!query) return

    const pathname = generateEntryPath(
      activeRepo,
      activeEntryInfo?.rev,
      '',
      'search'
    )

    router.push(`/files/${pathname}?q=${encodeURIComponent(query)}`)
  }

  return (
    <form
      onSubmit={onSubmit}
      className={cn(
        'flex w-full shrink-0 items-center bg-background px-4 py-3.5 transition duration-500 ease-in-out',
        className
      )}
    >
      <div className="relative w-full">
        <Input
          ref={inputRef}
          placeholder="Search for code..."
          className="w-full"
          autoComplete="off"
          value={query}
          onChange={e => setQuery(e.target.value)}
        />
        <div className="absolute right-2 top-0 flex h-full items-center">
          {query ? (
            <Button
              type="button"
              variant="ghost"
              size="icon"
              className="h-6 w-6 cursor-pointer"
              onClick={clearInput}
            >
              <IconClose />
            </Button>
          ) : (
            <kbd
              className="rounded-md border bg-secondary/50 px-1.5 text-xs leading-4 text-muted-foreground shadow-[inset_-0.5px_-1.5px_0_hsl(var(--muted))]"
              onClick={() => {
                inputRef.current?.focus()
              }}
            >
              s
            </kbd>
          )}
          <div className="ml-2 flex items-center border-l border-l-border pl-2">
            <Button
              variant="ghost"
              className="h-6 w-6 "
              size="icon"
              type="submit"
            >
              <IconSearch />
            </Button>
          </div>
        </div>
      </div>
    </form>
  )
}
