import { useContext, useMemo, useRef } from 'react'
import { zodResolver } from '@hookform/resolvers/zod'
import { useForm } from 'react-hook-form'
import * as z from 'zod'

import { useSelectedRepository } from '@/lib/hooks/use-repositories'
import { makeFormErrorHandler } from '@/lib/tabby/gql'
import { ExtendedCombinedError } from '@/lib/types'
import { isCodeSourceContext } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormMessage
} from '@/components/ui/form'
import { IconArrowRight } from '@/components/ui/icons'
import { Input } from '@/components/ui/input'
import { RepoSelect } from '@/components/textarea-search/repo-select'

import { PageContext } from './page-context'

export function NewPageForm({
  onSubmit
}: {
  onSubmit: (v: {
    titlePrompt: string
    codeSourceId: string | undefined
  }) => Promise<ExtendedCombinedError | void>
}) {
  const { fetchingContextInfo, contextInfo } = useContext(PageContext)
  const { selectedRepository, onSelectRepository } = useSelectedRepository()
  const inputRef = useRef<HTMLInputElement | null>(null)
  const formSchema = z.object({
    titlePrompt: z.string().trim()
  })

  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema)
  })
  const titlePrompt = form.watch('titlePrompt')

  const repos = useMemo(() => {
    return contextInfo?.sources.filter(x => isCodeSourceContext(x.sourceKind))
  }, [contextInfo?.sources])

  const focusInput = () => {
    inputRef.current?.focus()
  }

  const handleSelectRepo = (id: string | undefined) => {
    onSelectRepository(id)
    setTimeout(() => {
      focusInput()
    }, 10)
  }

  const handleSubmit = async (values: z.infer<typeof formSchema>) => {
    const error = await onSubmit({
      titlePrompt: values.titlePrompt.trim(),
      codeSourceId: selectedRepository?.id
    })

    if (error) {
      makeFormErrorHandler(form)(error)
    }
  }

  return (
    <Form {...form}>
      <form onSubmit={form.handleSubmit(handleSubmit)}>
        <div className="flex w-full items-center gap-2">
          <FormField
            control={form.control}
            name="titlePrompt"
            render={({ field: { ref, ...rest } }) => (
              <FormItem className="flex-1">
                <FormControl>
                  <Input
                    autoFocus
                    className="h-auto w-full border-none text-3xl font-semibold shadow-none outline-none focus-visible:ring-0"
                    placeholder="What is your page about?"
                    autoCapitalize="none"
                    autoComplete="off"
                    autoCorrect="off"
                    ref={e => {
                      ref(e)
                      inputRef.current = e
                    }}
                    {...rest}
                  />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
          <Button
            className="h-auto w-auto px-2"
            type="submit"
            disabled={!titlePrompt}
          >
            <IconArrowRight />
          </Button>
        </div>
        <div className="mt-4 pl-3" onClick={e => focusInput()}>
          <RepoSelect
            repos={repos}
            isInitializing={fetchingContextInfo}
            value={selectedRepository?.sourceId}
            onChange={handleSelectRepo}
            placeholder="Select repository"
            showChevron
          />
        </div>
        <div className="my-2">
          <FormMessage />
        </div>
      </form>
    </Form>
  )
}
