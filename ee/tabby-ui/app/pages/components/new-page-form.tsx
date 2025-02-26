import { zodResolver } from '@hookform/resolvers/zod'
import { useForm } from 'react-hook-form'
import * as z from 'zod'

import { makeFormErrorHandler } from '@/lib/tabby/gql'
import { ExtendedCombinedError } from '@/lib/types'
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

export function NewPageForm({
  onSubmit
}: {
  onSubmit: (title: string) => Promise<ExtendedCombinedError | void>
}) {
  const formSchema = z.object({
    title: z.string().trim()
  })
  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema)
  })
  const title = form.watch('title')

  const handleSubmit = async (values: z.infer<typeof formSchema>) => {
    const error = await onSubmit(values.title.trim())

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
            name="title"
            render={({ field }) => (
              <FormItem className="flex-1">
                <FormControl>
                  <Input
                    autoFocus
                    className="h-auto w-full border-none text-3xl font-semibold shadow-none outline-none focus-visible:ring-0"
                    placeholder="What is your page about?"
                    {...field}
                  />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
          <Button
            className="h-auto w-auto px-2"
            type="submit"
            disabled={!title}
          >
            <IconArrowRight />
          </Button>
        </div>
        <div className="my-2">
          <FormMessage />
        </div>
      </form>
    </Form>
  )
}
