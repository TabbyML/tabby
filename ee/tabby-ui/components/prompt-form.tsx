import { UseChatHelpers } from 'ai/react'
import * as React from 'react'
import { Button, buttonVariants } from '@/components/ui/button'
import { IconArrowElbow, IconEdit } from '@/components/ui/icons'
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger
} from '@/components/ui/tooltip'
import {
  Combobox,
  ComboboxAnchor,
  ComboboxContent,
  ComboboxOption,
  ComboboxTextarea
} from '@/components/ui/combobox'
import { Popover, PopoverAnchor, PopoverContent } from '@/components/ui/popover'
import { useEnterSubmit } from '@/lib/hooks/use-enter-submit'
import { cn, getSearchCompletionQueryName } from '@/lib/utils'
import useSWR from 'swr'
import fetcher from '@/lib/tabby-fetcher'
import { debounce, has } from 'lodash'
import { ISearchHit, SearchReponse } from '@/lib/types'

export interface PromptProps
  extends Pick<UseChatHelpers, 'input' | 'setInput'> {
  onSubmit: (value: string) => Promise<void>
  isLoading: boolean
}

export function PromptForm({
  onSubmit,
  input,
  setInput,
  isLoading
}: PromptProps) {
  const { formRef, onKeyDown } = useEnterSubmit()
  const [queryCompletionUrl, setQueryCompletionUrl] = React.useState<
    string | null
  >(null)
  const latestFetchKey = React.useRef('')

  useSWR<SearchReponse>(queryCompletionUrl, fetcher, {
    revalidateOnFocus: false,
    dedupingInterval: 500,
    onSuccess: (data, key) => {
      if (key !== latestFetchKey.current) return

      setOptions(data?.hits ?? [])
    }
  })

  const [options, setOptions] = React.useState<SearchReponse['hits']>([])
  const onSearch = React.useMemo(() => {
    return debounce((e: React.ChangeEvent<HTMLTextAreaElement>) => {
      const value = e.target?.value ?? ''
      const end = e.target?.selectionEnd ?? 0
      const queryname = getSearchCompletionQueryName(value, end)
      if (queryname) {
        const query = encodeURIComponent(`name:${queryname} kind:function`)
        const url = `/v1beta/search?q=${query}`
        latestFetchKey.current = url
        setQueryCompletionUrl(url)
      } else {
        setOptions([])
      }
    }, 200)
  }, [])

  const onSelectCompletion = (
    inputRef: React.MutableRefObject<
      HTMLTextAreaElement | HTMLInputElement | null
    >,
    item: ISearchHit
  ) => {
    const replaceString = '`@' + item?.doc?.name + '` '
    const selectionEnd = inputRef.current?.selectionEnd ?? 0
    const queryname = getSearchCompletionQueryName(input, selectionEnd)
    const prevInput = input
      .substring(0, selectionEnd)
      .replace(new RegExp(`@${queryname}$`), '')
    if (queryname) {
      setInput(prevInput + replaceString + input.substring(selectionEnd))
    }

    setOptions([])
  }

  return (
    <form
      onSubmit={async e => {
        e.preventDefault()
        if (!input?.trim()) {
          return
        }
        setInput('')
        await onSubmit(input)
      }}
      ref={formRef}
    >
      <Combobox options={options} onSelect={onSelectCompletion}>
        {({ open, inputRef, highlightedIndex }) => {
          return (
            <>
              <ComboboxAnchor>
                <div className="bg-background relative flex max-h-60 w-full grow flex-col overflow-hidden px-8 sm:rounded-md sm:border sm:px-12">
                  <span
                    className={cn(
                      buttonVariants({ size: 'sm', variant: 'ghost' }),
                      'bg-background hover:bg-background absolute left-0 top-4 h-8 w-8 rounded-full p-0 sm:left-4'
                    )}
                  >
                    <IconEdit />
                  </span>
                  <ComboboxTextarea
                    tabIndex={0}
                    rows={1}
                    placeholder="Ask a question."
                    spellCheck={false}
                    className="min-h-[60px] w-full resize-none bg-transparent px-4 py-[1.3rem] focus-within:outline-none sm:text-sm"
                    value={input}
                    ref={
                      inputRef as React.MutableRefObject<HTMLTextAreaElement>
                    }
                    onChange={e => {
                      if (has(e, 'target.value')) {
                        let event = e as React.ChangeEvent<HTMLTextAreaElement>
                        setInput(event.target.value)
                        onSearch?.(event)
                      }
                    }}
                    onKeyDown={onKeyDown}
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
              </ComboboxAnchor>
              <ComboboxContent
                align="start"
                onOpenAutoFocus={e => e.preventDefault()}
                // popupMatchAnchorWidth
                className="w-1/2 max-w-xl"
              >
                <Popover open={!!options?.[highlightedIndex]}>
                  <PopoverAnchor>
                    <div className="max-h-[300px] overflow-y-scroll">
                      {open &&
                        !!options?.length &&
                        options.map((item, index) => (
                          <ComboboxOption
                            item={item}
                            index={index}
                            key={item?.id}
                          >
                            <div className="flex flex-col overflow-x-hidden">
                              <div className="truncate">{item?.doc?.name}</div>
                              <div className="text-muted-foreground truncate text-sm">
                                {item?.doc?.body}
                              </div>
                            </div>
                          </ComboboxOption>
                        ))}
                    </div>
                  </PopoverAnchor>
                  <PopoverContent
                    asChild
                    align="end"
                    side="right"
                    alignOffset={-4}
                    onOpenAutoFocus={e => e.preventDefault()}
                    onFocus={e => e.preventDefault()}
                    onClick={e => e.preventDefault()}
                    className="max-w-xl rounded-none"
                  >
                    <div className="flex flex-col px-2">
                      <div className="mb-2">
                        {options?.[highlightedIndex]?.doc?.name}
                      </div>
                      <div className="text-muted-foreground flex-1 overflow-auto whitespace-pre">
                        {options?.[highlightedIndex]?.doc?.body}
                      </div>
                    </div>
                  </PopoverContent>
                </Popover>
              </ComboboxContent>
            </>
          )
        }}
      </Combobox>
    </form>
  )
}
