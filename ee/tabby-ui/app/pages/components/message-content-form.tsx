import { zodResolver } from '@hookform/resolvers/zod'
import { useForm } from 'react-hook-form'
import Textarea from 'react-textarea-autosize'
import * as z from 'zod'

import { makeFormErrorHandler } from '@/lib/tabby/gql'
import { ExtendedCombinedError } from '@/lib/types'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormMessage
} from '@/components/ui/form'
import { IconSpinner } from '@/components/ui/icons'

interface Props {
  message: string
  onCancel: () => void
  onSubmit: (newMessage: string) => Promise<ExtendedCombinedError | void>
  inputClassName?: string
}

export function MessageContentForm({
  message,
  onCancel,
  onSubmit,
  inputClassName
}: Props) {
  const formSchema = z.object({
    content: z.string().trim()
  })
  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: { content: message }
  })
  const { isSubmitting } = form.formState

  const handleSubmit = async (values: z.infer<typeof formSchema>) => {
    const error = await onSubmit(values.content)

    if (error) {
      makeFormErrorHandler(form)(error)
    }
  }

  return (
    <Form {...form}>
      <form onSubmit={form.handleSubmit(handleSubmit)}>
        <FormField
          control={form.control}
          name="content"
          render={({ field }) => (
            <FormItem>
              <FormControl>
                <Textarea
                  autoFocus
                  autoCapitalize="off"
                  autoComplete="off"
                  autoCorrect="off"
                  minRows={2}
                  maxRows={20}
                  className={cn(
                    'w-full rounded-lg border bg-background p-4 outline-ring',
                    inputClassName
                  )}
                  {...field}
                />
              </FormControl>
              <FormMessage />
            </FormItem>
          )}
        />
        <div className="mb-3 mt-1 flex items-center justify-between gap-2 px-2">
          <div>
            <FormMessage />
          </div>
          <div className="flex items-center gap-2">
            <Button
              type="button"
              variant="outline"
              onClick={onCancel}
              className="min-w-[2rem]"
            >
              Cancel
            </Button>
            <Button type="submit" disabled={isSubmitting}>
              {isSubmitting && (
                <IconSpinner className="mr-2 h-4 w-4 animate-spin" />
              )}
              Save
            </Button>
          </div>
        </div>
      </form>
    </Form>
  )
}
