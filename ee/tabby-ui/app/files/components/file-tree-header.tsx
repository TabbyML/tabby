'use client'

import React, { useContext } from 'react'

import { useDebounceCallback } from '@/lib/hooks/use-debounce'
import useRouterStuff from '@/lib/hooks/use-router-stuff'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import {
  Combobox,
  ComboboxAnchor,
  ComboboxContent,
  ComboboxInput,
  ComboboxOption
} from '@/components/ui/combobox'
import {
  IconClose,
  IconDirectorySolid,
  IconFile,
  IconFolderGit
} from '@/components/ui/icons'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from '@/components/ui/select'

import { SourceCodeBrowserContext } from './source-code-browser'
import { resolveRepoNameFromPath } from './utils'

interface FileTreeHeaderProps extends React.HTMLAttributes<HTMLDivElement> {}

type SearchOption = { entry: string; type: 'dir' | 'file'; id: string }

const FileTreeHeader: React.FC<FileTreeHeaderProps> = ({
  className,
  ...props
}) => {
  const { activePath, fileTreeData, setActivePath, initialized } = useContext(
    SourceCodeBrowserContext
  )
  const { updateSearchParams } = useRouterStuff()
  const curerntRepoName = resolveRepoNameFromPath(activePath)

  const inputRef = React.useRef<HTMLInputElement>(null)
  const ignoreFetchResultRef = React.useRef(false)
  const [input, setInput] = React.useState<string>()
  const [searchFileter, setSearchFilter] = React.useState<string>()
  const [options, setOptions] = React.useState<Array<SearchOption>>()
  const [optionsVisible, setOptionsVisible] = React.useState(false)

  const repoName = resolveRepoNameFromPath(activePath)

  const noIndexedRepo = initialized && !fileTreeData?.length

  const onSelectRepo = (name: string) => {
    setActivePath(name)
  }

  const memoizedMatchedIndices = React.useMemo(() => {
    return options?.map(option =>
      searchFileter ? getMatchedIndices(searchFileter, option.entry) : []
    )
  }, [options, searchFileter])

  const onInputValueChange = useDebounceCallback((v: string | undefined) => {
    if (!v) {
      ignoreFetchResultRef.current = true
      setOptions([])
      setOptionsVisible(false)
    } else {
      ignoreFetchResultRef.current = false
      // if (v === 'test-not-found') {
      //   ignoreFetchResultRef.current = true
      //   setOptions([])
      //   setOptionsVisible(true)
      // }
      setTimeout(() => {
        if (!ignoreFetchResultRef.current) {
          let ops: SearchOption[] = [
            { entry: 'test1', type: 'file', id: 'test1' },
            { entry: 'test2', type: 'file', id: 'test2' },
            { entry: 'path/to/test', type: 'dir', id: 'path/to/test' }
          ]
          setSearchFilter(v)
          setOptions(ops)
          setOptionsVisible(true)
        }
      }, 100)
    }
  }, 300)

  const onClearInput = () => {
    onInputValueChange.run('')
    onInputValueChange.flush()
  }

  const onSelectFile = (value: SearchOption) => {
    // todo fetch dirs and then update activePath, or implement this logic in code borwser fetcher
    console.log(value)
  }

  // shortcut 't'
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

      if (event.key === 't') {
        event.preventDefault()
        inputRef.current?.focus()
      }
    }

    window.addEventListener('keydown', handleKeyDown)

    return () => {
      window.removeEventListener('keydown', handleKeyDown)
    }
  }, [])

  return (
    <div className={cn(className)} {...props}>
      <div className="py-4 font-bold leading-8">Files</div>
      <div className="space-y-3">
        <Select onValueChange={onSelectRepo} value={repoName}>
          <SelectTrigger>
            <SelectValue>
              <div className="flex items-center gap-2">
                <IconFolderGit />
                <span className={repoName ? '' : 'text-muted-foreground'}>
                  {repoName || 'Pick a repository'}
                </span>
              </div>
            </SelectValue>
          </SelectTrigger>
          <SelectContent>
            {noIndexedRepo ? (
              <SelectItem isPlaceHolder value="" disabled>
                No indexed repository
              </SelectItem>
            ) : (
              <>
                {fileTreeData?.map(repo => {
                  return (
                    <SelectItem key={repo.fullPath} value={repo.name}>
                      {repo.name}
                    </SelectItem>
                  )
                })}
              </>
            )}
          </SelectContent>
        </Select>
        <Combobox
          options={options}
          onSelect={onSelectFile}
          open={optionsVisible}
          onOpenChange={v => {
            if ((input || options?.length) && v) {
              setOptionsVisible(true)
            } else {
              setOptionsVisible(false)
            }
          }}
        >
          {({ highlightedIndex }) => {
            return (
              <>
                <ComboboxAnchor>
                  <div className="relative">
                    <ComboboxInput
                      className="pr-8"
                      placeholder="Go to file"
                      spellCheck={false}
                      value={input}
                      ref={inputRef}
                      onChange={e => {
                        let value = e.target.value
                        setInput(value)
                        if (!value) {
                          onClearInput()
                        } else {
                          onInputValueChange.run(value)
                        }
                      }}
                    />
                    <div className="absolute right-2 top-0 flex h-full items-center">
                      {input ? (
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-6 w-6 cursor-pointer"
                          onClick={e => {
                            setInput('')
                            onClearInput()
                            inputRef.current?.focus()
                          }}
                        >
                          <IconClose />
                        </Button>
                      ) : (
                        <kbd
                          className="rounded-md border bg-secondary/50 px-1.5 text-xs leading-4 text-muted-foreground shadow-[inset_-0.5px_-1.5px_0_hsl(var(--muted))]"
                          onClick={e => {
                            inputRef.current?.focus()
                          }}
                        >
                          t
                        </kbd>
                      )}
                    </div>
                  </div>
                </ComboboxAnchor>
                <ComboboxContent
                  align="start"
                  side="bottom"
                  onOpenAutoFocus={e => e.preventDefault()}
                  className="max-h-[50vh] min-w-[30vw] overflow-y-scroll"
                >
                  <>
                    {options?.length ? (
                      options?.map((item, index) => (
                        <ComboboxOption
                          item={item}
                          index={index}
                          key={item?.id}
                          className="flex w-full items-center gap-2 overflow-x-hidden"
                        >
                          <div className="shrink-0">
                            {item.type === 'dir' ? (
                              <IconDirectorySolid
                                style={{ color: 'rgb(84, 174, 255)' }}
                              />
                            ) : (
                              <IconFile />
                            )}
                          </div>
                          <div className="flex-1 truncate">
                            <HighlightMatches
                              matchedIndices={
                                memoizedMatchedIndices?.[index] ?? []
                              }
                            >
                              {item.entry}
                            </HighlightMatches>
                          </div>
                          {highlightedIndex === index && (
                            <div className="shrink-0`">
                              {item.type === 'dir'
                                ? 'Go to folder'
                                : 'Go to file'}
                            </div>
                          )}
                        </ComboboxOption>
                      ))
                    ) : (
                      <div className="flex h-16 items-center justify-center">
                        No matches found
                      </div>
                    )}
                  </>
                </ComboboxContent>
              </>
            )
          }}
        </Combobox>
      </div>
    </div>
  )
}

const HighlightMatches = React.memo(
  ({
    children,
    matchedIndices
  }: {
    children: string
    matchedIndices: number[]
  }) => {
    let lastIndex = 0
    const highlightedText = []

    matchedIndices.forEach(index => {
      // add non-highlighted text
      highlightedText.push(
        <span key={`text-${lastIndex}`}>
          {children.substring(lastIndex, index)}
        </span>
      )
      // add highlighted text
      highlightedText.push(
        <span key={`match-${index}`} className="font-bold text-foreground">
          {children.substring(index, index + 1)}
        </span>
      )
      lastIndex = index + 1
    })

    highlightedText.push(
      <span key={`text-${lastIndex}`}>{children.substring(lastIndex)}</span>
    )

    return <p className="text-muted-foreground">{highlightedText}</p>
  }
)
HighlightMatches.displayName = 'HighlightMatches'

function getMatchedIndices(s: string, t: string) {
  let result: number[] = []
  let p = 0
  for (let i = 0; i < t.length; i++) {
    if (s[p] === t[i]) {
      result.push(i)
      p++
    }
  }
  return result?.length === s.length ? result : []
}

export { FileTreeHeader }
