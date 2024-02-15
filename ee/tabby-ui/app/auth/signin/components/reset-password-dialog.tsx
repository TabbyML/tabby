'use client'

import React from 'react'
import { zodResolver } from '@hookform/resolvers/zod'
import { useForm } from 'react-hook-form'
import { toast } from 'sonner'
import * as z from 'zod'

import { PLACEHOLDER_EMAIL_FORM } from '@/lib/constants'
import { graphql } from '@/lib/gql/generates/gql'
import { useMutation } from '@/lib/tabby/gql'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger
} from '@/components/ui/dialog'
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage
} from '@/components/ui/form'
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
interface ResetPasswordDialogProps {
  children: React.ReactNode
}

export const ResetPasswordDialog: React.FC<ResetPasswordDialogProps> = ({
  children
}) => {
  const form = useForm<FormValeus>({
    resolver: zodResolver(formSchema)
  })

  const requestPasswordResetEmail = useMutation(
    requestPasswordResetEmailMutation,
    { form }
  )

  const onSubmit = (values: FormValeus) => {
    requestPasswordResetEmail({
      input: values
    }).then(res => {
      if (res?.data?.requestPasswordResetEmail) {
        toast.success('email sent')
      }
    })
  }

  return (
    <Dialog>
      <DialogTrigger asChild>{children}</DialogTrigger>
      <DialogContent>
        <DialogHeader className="gap-3">
          <DialogTitle>Reset Password</DialogTitle>
          <DialogDescription>
            Enter your email address and receive an email to reset password.
          </DialogDescription>
        </DialogHeader>
        <Form {...form}>
          <form className="grid gap-2" onSubmit={form.handleSubmit(onSubmit)}>
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
            <Button type="submit" className="mt-1">
              Send Email
            </Button>
          </form>
          <FormMessage className="text-center" />
        </Form>
      </DialogContent>
    </Dialog>
  )
}
