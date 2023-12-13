'use client'

import * as React from 'react'
import { zodResolver } from '@hookform/resolvers/zod'
import { useForm } from 'react-hook-form'
import * as z from 'zod'

import { graphql } from '@/lib/gql/generates'
import { useMutation } from '@/lib/tabby/gql'
import { Button } from '@/components/ui/button'
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormMessage
} from '@/components/ui/form'
import { Input } from '@/components/ui/input'

const createInvitationMutation = graphql(/* GraphQL */ `
  mutation CreateInvitation($email: String!) {
    createInvitation(email: $email)
  }
`)

const formSchema = z.object({
  email: z.string().email('Invalid email address')
})

export default function CreateInvitationForm({
  onCreated
}: {
  onCreated: () => void
}) {
  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema)
  })

  const { isSubmitting } = form.formState
  const createInvitation = useMutation(createInvitationMutation, {
    onCompleted() {
      form.reset({ email: '' })
      onCreated()
    },
    form
  })

  return (
    <Form {...form}>
      <div className="flex flex-col items-start gap-2">
        <form
          className="flex w-full items-center gap-2"
          onSubmit={form.handleSubmit(createInvitation)}
        >
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
                    {...field}
                  />
                </FormControl>
              </FormItem>
            )}
          />
          <Button type="submit" disabled={isSubmitting}>
            Invite
          </Button>
        </form>
        <FormMessage className="text-center" />
      </div>
    </Form>
  )
}
