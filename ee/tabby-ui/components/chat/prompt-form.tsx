import * as React from 'react'
import { UseChatHelpers } from 'ai/react'
import { debounce, has } from 'lodash-es'
import useSWR from 'swr'

import { useEnterSubmit } from '@/lib/hooks/use-enter-submit'
import fetcher from '@/lib/tabby/fetcher'
import type { ISearchHit, SearchReponse } from '@/lib/types'
import { cn } from '@/lib/utils'
import { Button, buttonVariants } from '@/components/ui/button'
import {
  IconArrowElbow,
  IconEdit,
  IconSymbolFunction
} from '@/components/ui/icons'
import { Popover, PopoverAnchor, PopoverContent } from '@/components/ui/popover'
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger
} from '@/components/ui/tooltip'
import {
  SearchableSelect,
  SearchableSelectAnchor,
  SearchableSelectContent,
  SearchableSelectOption,
  SearchableSelectTextarea
} from '@/components/searchable-select'

export interface PromptProps
  extends Pick<UseChatHelpers, 'input' | 'setInput'> {
  onSubmit: (value: string) => Promise<void>
  isLoading: boolean
  chatInputRef: React.RefObject<HTMLTextAreaElement>
}

export interface PromptFormRef {
  focus: () => void
}

function PromptFormRenderer(
  { onSubmit, input, setInput, isLoading, chatInputRef }: PromptProps,
  ref: React.ForwardedRef<PromptFormRef>
) {
  const { formRef, onKeyDown } = useEnterSubmit()
  const [queryCompletionUrl, setQueryCompletionUrl] = React.useState<
    string | null
  >(null)
  const [suggestionOpen, setSuggestionOpen] = React.useState(false)
  // store the input selection for replacing inputValue
  const prevInputSelectionEnd = React.useRef<number>()
  // for updating the input selection after replacing
  const nextInputSelectionRange = React.useRef<[number, number]>()
  const [options, setOptions] = React.useState<SearchReponse['hits']>([])
  const [selectedCompletionsMap, setSelectedCompletionsMap] = React.useState<
    Record<string, ISearchHit>
  >({})

  const { data: completionData } = useSWR<SearchReponse>(
    queryCompletionUrl,
    fetcher,
    {
      revalidateOnFocus: false,
      dedupingInterval: 0,
      errorRetryCount: 0
    }
  )

  React.useEffect(() => {
    const suggestions = completionData?.hits ?? []
    setOptions(suggestions)
    setSuggestionOpen(!!suggestions?.length)
  }, [completionData?.hits])

  React.useImperativeHandle(ref, () => ({
    focus: () => chatInputRef.current?.focus()
  }))

  React.useEffect(() => {
    if (
      input &&
      chatInputRef.current &&
      chatInputRef.current !== document.activeElement
    ) {
      chatInputRef.current.focus()
    }
  }, [input, chatInputRef])

  React.useLayoutEffect(() => {
    if (nextInputSelectionRange.current?.length) {
      chatInputRef.current?.setSelectionRange?.(
        nextInputSelectionRange.current[0],
        nextInputSelectionRange.current[1]
      )
      nextInputSelectionRange.current = undefined
    }
  }, [chatInputRef])

  const handleSearchCompletion = React.useMemo(() => {
    return debounce((e: React.ChangeEvent<HTMLTextAreaElement>) => {
      const value = e.target?.value ?? ''
      const end = e.target?.selectionEnd ?? 0
      const queryNameMatches = getSearchCompletionQueryName(value, end)
      const queryName = queryNameMatches?.[1]
      if (queryName) {
        const query = encodeURIComponent(`name:${queryName} AND kind:function`)
        const url = `/v1beta/search?q=${query}`
        setQueryCompletionUrl(url)
      } else {
        setOptions([])
        setSuggestionOpen(false)
      }
    }, 200)
  }, [])

  const handleCompletionSelect = (item: ISearchHit) => {
    const selectionEnd = prevInputSelectionEnd.current ?? 0
    const queryNameMatches = getSearchCompletionQueryName(input, selectionEnd)
    if (queryNameMatches) {
      setSelectedCompletionsMap({
        ...selectedCompletionsMap,
        [`@${item.doc?.name}`]: item
      })
      const replaceString = `@${item?.doc?.name} `
      const prevInput = input
        .substring(0, selectionEnd)
        .replace(new RegExp(queryNameMatches[0]), '')
      const nextSelectionEnd = prevInput.length + replaceString.length
      nextInputSelectionRange.current = [nextSelectionEnd, nextSelectionEnd]
      setInput(prevInput + replaceString + input.slice(selectionEnd))
    }
    setOptions([])
    setSuggestionOpen(false)
  }

  const handlePromptSubmit: React.FormEventHandler<
    HTMLFormElement
  > = async e => {
    e.preventDefault()
    if (!input?.trim() || isLoading) {
      return
    }

    let finalInput = input
    Object.keys(selectedCompletionsMap).forEach(key => {
      const completion = selectedCompletionsMap[key]
      if (!completion?.doc) return
      finalInput = finalInput.replaceAll(
        key,
        `\n${'```'}${completion.doc?.language ?? ''}\n${
          completion.doc.body ?? ''
        }\n${'```'}\n`
      )
    })

    setInput('')
    await onSubmit(finalInput)
  }

  const handleTextareaKeyDown = (
    e: React.KeyboardEvent<HTMLTextAreaElement>,
    isOpen: boolean
  ) => {
    if (e.key === 'Enter' && isOpen) {
      e.preventDefault()
    } else if (
      isOpen &&
      ['ArrowRight', 'ArrowLeft', 'Home', 'End'].includes(e.key)
    ) {
      setOptions([])
      setSuggestionOpen(false)
    } else {
      if (!isOpen) {
        ;(e as any).preventDownshiftDefault = true
      }
      onKeyDown(e)
    }
  }

  return (
    <form onSubmit={handlePromptSubmit} ref={formRef}>
      <SearchableSelect
        options={options}
        onSelect={handleCompletionSelect}
        open={suggestionOpen}
        onOpenChange={isOpen => {
          if (isOpen && options?.length) {
            setSuggestionOpen(isOpen)
          } else {
            setSuggestionOpen(false)
            setOptions([])
          }
        }}
      >
        {({ open, highlightedIndex }) => {
          const highlightedOption = options?.[highlightedIndex]

          return (
            <>
              <SearchableSelectAnchor>
                <div className="relative flex max-h-60 w-full grow flex-col overflow-hidden bg-background px-8 sm:rounded-md sm:border sm:px-12">
                  <span
                    className={cn(
                      buttonVariants({ size: 'sm', variant: 'ghost' }),
                      'absolute left-0 top-4 h-8 w-8 rounded-full bg-background p-0 hover:bg-background sm:left-4'
                    )}
                  >
                    <IconEdit />
                  </span>
                  <SearchableSelectTextarea
                    tabIndex={0}
                    rows={1}
                    placeholder="Ask a question."
                    spellCheck={false}
                    className="min-h-[60px] w-full resize-none bg-transparent px-4 py-[1.3rem] focus-within:outline-none"
                    value={input}
                    ref={chatInputRef}
                    onChange={e => {
                      if (has(e, 'target.value')) {
                        prevInputSelectionEnd.current = e.target.selectionEnd
                        setInput(e.target.value)
                        // TODO: Temporarily disabling the current search function. Will be replaced with a different search functionality in the future.
                        // handleSearchCompletion(e)
                      } else {
                        prevInputSelectionEnd.current = undefined
                      }
                    }}
                    onKeyDown={e => handleTextareaKeyDown(e, open)}
                  />
                  <div className="absolute right-0 top-4 sm:right-4">
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Button
                          type="submit"
                          size="icon"
                          disabled={isLoading || input === ''}
                        >
                          <IconArrowElbow />
                          <span className="sr-only">Send message</span>
                        </Button>
                      </TooltipTrigger>
                      <TooltipContent>Send message</TooltipContent>
                    </Tooltip>
                  </div>
                </div>
              </SearchableSelectAnchor>
              <SearchableSelectContent
                align="start"
                side="top"
                onOpenAutoFocus={e => e.preventDefault()}
                className="w-[60vw] md:w-[430px]"
              >
                <Popover open={open && !!highlightedOption}>
                  <PopoverAnchor asChild>
                    <div className="max-h-[300px] overflow-y-scroll">
                      {open &&
                        !!options?.length &&
                        options.map((item, index) => (
                          <SearchableSelectOption
                            item={item}
                            index={index}
                            key={item?.id}
                          >
                            <div className="flex w-full items-center justify-between gap-8 overflow-x-hidden">
                              <div className="flex items-center gap-1">
                                <IconForCompletionKind kind={item?.doc?.kind} />
                                <div className="max-w-[200px] truncate">
                                  {item?.doc?.name}(...)
                                </div>
                              </div>
                              <div className="flex-1 truncate text-right text-sm text-muted-foreground">
                                {item?.doc?.body}
                              </div>
                            </div>
                          </SearchableSelectOption>
                        ))}
                    </div>
                  </PopoverAnchor>
                  <PopoverContent
                    asChild
                    align="start"
                    side="right"
                    alignOffset={-4}
                    onOpenAutoFocus={e => e.preventDefault()}
                    onKeyDownCapture={e => e.preventDefault()}
                    className="rounded-none"
                    collisionPadding={{ bottom: 120 }}
                  >
                    <div className="flex max-h-[70vh] w-[20vw] flex-col overflow-y-auto px-2 md:w-[240px] lg:w-[340px]">
                      <div className="mb-2">
                        {highlightedOption?.doc?.kind
                          ? `(${highlightedOption?.doc?.kind}) `
                          : ''}
                        {highlightedOption?.doc?.name}
                      </div>
                      <div className="flex-1 whitespace-pre-wrap break-all text-muted-foreground">
                        {highlightedOption?.doc?.body}
                      </div>
                    </div>
                  </PopoverContent>
                </Popover>
              </SearchableSelectContent>
            </>
          )
        }}
      </SearchableSelect>
    </form>
  )
}

export const PromptForm = React.forwardRef<PromptFormRef, PromptProps>(
  PromptFormRenderer
)

/**
 * Retrieves the name of the completion query from a given string@.
 * @param {string} val - The input string to search for the completion query name.
 * @param {number | undefined} selectionEnd - The index at which the selection ends in the input string.
 * @return {string | undefined} - The name of the completion query if found, otherwise undefined.
 */
export function getSearchCompletionQueryName(
  val: string,
  selectionEnd: number | undefined
): RegExpExecArray | null {
  const queryString = val.substring(0, selectionEnd)
  const matches = /@(\w+)$/.exec(queryString)
  return matches
}

function IconForCompletionKind({
  kind,
  ...rest
}: { kind: string | undefined } & React.ComponentProps<'svg'>) {
  switch (kind) {
    case 'function':
      return <IconSymbolFunction {...rest} />
    default:
      return <IconSymbolFunction {...rest} />
  }
}
