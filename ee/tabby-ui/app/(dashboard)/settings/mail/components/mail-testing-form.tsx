'use client'

import * as React from 'react'
import { zodResolver } from '@hookform/resolvers/zod'
import { useForm } from 'react-hook-form'
import * as z from 'zod'

import { Button } from '@/components/ui/button'
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormMessage
} from '@/components/ui/form'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'

const formSchema = z.object({
  email: z.string().email('Invalid email address')
})

export default function MailTestingForm({
  onSendTest
}: {
  onSendTest: () => Promise<any>
}) {
  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema)
  })

  const { isSubmitting } = form.formState
  const handleSubmitTest = () => {
    onSendTest()
  }

  return (
    <Form {...form}>
      <div className="flex flex-col items-start gap-2">
        <form
          className="flex flex-col gap-2"
          onSubmit={form.handleSubmit(handleSubmitTest)}
        >
          <Label>Send Test Email To</Label>
          <div className="flex gap-4">
            <FormField
              control={form.control}
              name="email"
              render={({ field }) => (
                <FormItem>
                  <FormControl>
                    <Input
                      placeholder="Email"
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
              Send
            </Button>
          </div>
        </form>
        <FormMessage className="text-center" />
      </div>
    </Form>
  )
}
