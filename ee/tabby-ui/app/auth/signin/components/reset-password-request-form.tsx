'use client'

import React from 'react'
import { zodResolver } from '@hookform/resolvers/zod'
import { useForm } from 'react-hook-form'
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
  FormLabel,
  FormMessage
} from '@/components/ui/form'
import { IconSpinner } from '@/components/ui/icons'
import { Input } from '@/components/ui/input'

const requestPasswordResetEmailMutation = graphql(/* GraphQL */ `
  mutation requestPasswordResetEmail($input: RequestPasswordResetEmailInput!) {
    requestPasswordResetEmail(input: $input)
  }
`)

const formSchema = z.object({
  email: z.string().email('Invalid email address')
})

type FormValeus = z.infer<typeof formSchema>
interface ResetPasswordRequestFormProps {
  onSuccess?: (email: string) => void
}

export const ResetPasswordRequestForm: React.FC<
  ResetPasswordRequestFormProps
> = ({ onSuccess }) => {
  const form = useForm<FormValeus>({
    resolver: zodResolver(formSchema)
  })
  const { isSubmitting } = form.formState

  const requestPasswordResetEmail = useMutation(
    requestPasswordResetEmailMutation,
    { form }
  )

  const onSubmit = (values: FormValeus) => {
    return requestPasswordResetEmail({
      input: values
    }).then(res => {
      if (res?.data?.requestPasswordResetEmail) {
        onSuccess?.(values.email)
      }
    })
  }

  return (
    <Form {...form}>
      <div className="grid gap-2">
        <form className="grid gap-4" onSubmit={form.handleSubmit(onSubmit)}>
          <FormField
            control={form.control}
            name="email"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Email</FormLabel>
                <FormControl>
                  <Input
                    placeholder={PLACEHOLDER_EMAIL_FORM}
                    type="email"
                    autoCapitalize="none"
                    autoComplete="email"
                    autoCorrect="off"
                    {...field}
                  />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
          <Button type="submit" className="mt-2">
            {isSubmitting && (
              <IconSpinner className="mr-2 h-4 w-4 animate-spin" />
            )}
            Send Email
          </Button>
        </form>
        <FormMessage className="text-center" />
      </div>
    </Form>
  )
}
