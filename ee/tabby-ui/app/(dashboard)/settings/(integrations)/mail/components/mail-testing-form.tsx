'use client'

import * as React from 'react'
import { zodResolver } from '@hookform/resolvers/zod'
import { useForm } from 'react-hook-form'
import { toast } from 'sonner'
import * as z from 'zod'

import { PLACEHOLDER_EMAIL_FORM } from '@/lib/constants'
import { graphql } from '@/lib/gql/generates/gql'
import { useMutation } from '@/lib/tabby/gql'
import { Button } from '@/components/ui/button'
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormMessage
} from '@/components/ui/form'
import { IconSpinner } from '@/components/ui/icons'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'

const sendTestEmailMutation = graphql(/* GraphQL */ `
  mutation SendTestEmail($to: String!) {
    sendTestEmail(to: $to)
  }
`)

const formSchema = z.object({
  to: z.string().email('Invalid email address')
})

type FormValues = z.infer<typeof formSchema>

export default function MailTestingForm({
  onSuccess
}: {
  onSuccess?: () => Promise<any>
}) {
  const form = useForm<FormValues>({
    resolver: zodResolver(formSchema)
  })

  const { isSubmitting } = form.formState
  const sendTestEmail = useMutation(sendTestEmailMutation, { form })
  const onSubmit = (values: FormValues) => {
    return sendTestEmail(values).then(res => {
      if (res?.data?.sendTestEmail) {
        toast.success(
          'A test email has been sent, please check your inbox to verify.'
        )
        onSuccess?.()
      }
    })
  }

  return (
    <Form {...form}>
      <div className="flex flex-col items-start gap-2">
        <form
          className="flex flex-col gap-2"
          onSubmit={form.handleSubmit(onSubmit)}
        >
          <Label>Send Test Email To</Label>
          <div className="flex gap-4">
            <FormField
              control={form.control}
              name="to"
              render={({ field }) => (
                <FormItem>
                  <FormControl>
                    <Input
                      placeholder={`e.g. ${PLACEHOLDER_EMAIL_FORM}`}
                      type="email"
                      autoCapitalize="none"
                      autoComplete="email"
                      autoCorrect="off"
                      className="w-80"
                      {...field}
                    />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            <Button
              className="self-start"
              type="submit"
              disabled={isSubmitting}
            >
              {isSubmitting && (
                <IconSpinner className="mr-2 h-4 w-4 animate-spin" />
              )}
              Send
            </Button>
          </div>
        </form>
        <FormMessage className="text-center" />
      </div>
    </Form>
  )
}
