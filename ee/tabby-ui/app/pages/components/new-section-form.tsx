import { zodResolver } from '@hookform/resolvers/zod'
import { useForm } from 'react-hook-form'
import Textarea from 'react-textarea-autosize'
import * as z from 'zod'

import { useEnterSubmit } from '@/lib/hooks/use-enter-submit'
import { makeFormErrorHandler } from '@/lib/tabby/gql'
import { ExtendedCombinedError } from '@/lib/types'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Form, FormControl, FormField, FormItem } from '@/components/ui/form'
import { IconPlus } from '@/components/ui/icons'

export function NewSectionForm({
  onSubmit,
  disabled: propDisabled,
  className
}: {
  onSubmit: (title: string) => Promise<ExtendedCombinedError | void>
  disabled?: boolean
  className?: string
}) {
  const formSchema = z.object({
    title: z.string().trim()
  })
  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema)
  })

  const { formRef, onKeyDown } = useEnterSubmit()

  const title = form.watch('title')
  const { isSubmitting } = form.formState

  const disabled = propDisabled || !title || isSubmitting

  const handleSubmit = async (values: z.infer<typeof formSchema>) => {
    if (disabled) return
    const title = values.title
    onSubmit(title).then(error => {
      if (error) {
        makeFormErrorHandler(form)(error)
      }
    })
    form.reset({
      title: ''
    })
  }

  return (
    <Form {...form}>
      <form
        className={cn(className)}
        onSubmit={form.handleSubmit(handleSubmit)}
        ref={formRef}
      >
        <div className="relative">
          <FormField
            control={form.control}
            name="title"
            render={({ field }) => (
              <FormItem className="flex-1">
                <FormControl>
                  <Textarea
                    draggable={false}
                    minRows={2}
                    maxRows={6}
                    className="w-full rounded-lg border-2 bg-background p-4 pr-12 text-xl outline-ring"
                    placeholder="What is the section about?"
                    onKeyDown={onKeyDown}
                    autoCapitalize="off"
                    autoComplete="off"
                    autoCorrect="off"
                    {...field}
                  />
                </FormControl>
              </FormItem>
            )}
          />
          <Button
            size="icon"
            className="absolute right-4 top-1/2 z-10 h-7 w-7 -translate-y-1/2 rounded-full"
            type="submit"
            disabled={disabled}
          >
            <IconPlus />
          </Button>
        </div>
      </form>
    </Form>
  )
}
