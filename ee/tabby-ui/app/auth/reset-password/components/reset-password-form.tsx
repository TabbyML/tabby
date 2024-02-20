'use client'

import * as React from 'react'
import { useRouter } from 'next/navigation'
import { zodResolver } from '@hookform/resolvers/zod'
import { useForm } from 'react-hook-form'
import { toast } from 'sonner'
import * as z from 'zod'

import { graphql } from '@/lib/gql/generates'
import { useMutation } from '@/lib/tabby/gql'
import { cn } from '@/lib/utils'
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

const passwordResetMutation = graphql(/* GraphQL */ `
  mutation passwordReset($input: PasswordResetInput!) {
    passwordReset(input: $input)
  }
`)

const formSchema = z.object({
  password1: z.string(),
  password2: z.string(),
  code: z.string().optional()
})

type FormValues = z.infer<typeof formSchema>

interface ResetPasswordFormProps extends React.HTMLAttributes<HTMLDivElement> {
  code?: string
}

export function ResetPasswordForm({
  className,
  code,
  ...props
}: ResetPasswordFormProps) {
  const form = useForm<FormValues>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      code
    }
  })

  const router = useRouter()
  const { isSubmitting } = form.formState

  const passwordReset = useMutation(passwordResetMutation, {
    form
  })

  const onSubmit = (values: FormValues) => {
    passwordReset({
      input: {
        ...values,
        code: values.code ?? ''
      }
    }).then(res => {
      if (res?.data?.passwordReset) {
        toast.success('Password reset successful')
        setTimeout(() => {
          router.replace('/auth/signin')
        }, 1000)
      }
    })
  }

  return (
    <div className={cn('grid gap-6', className)} {...props}>
      <Form {...form}>
        <form className="grid gap-2" onSubmit={form.handleSubmit(onSubmit)}>
          <FormField
            control={form.control}
            name="password1"
            render={({ field }) => (
              <FormItem>
                <FormLabel>New Password</FormLabel>
                <FormControl>
                  <Input type="password" {...field} />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
          <FormField
            control={form.control}
            name="password2"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Confirm New Password</FormLabel>
                <FormControl>
                  <Input type="password" {...field} />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
          <FormField
            control={form.control}
            name="code"
            render={({ field }) => (
              <FormItem className="hidden">
                <FormControl>
                  <Input type="hidden" {...field} />
                </FormControl>
              </FormItem>
            )}
          />
          <Button type="submit" className="mt-1" disabled={isSubmitting}>
            {isSubmitting && (
              <IconSpinner className="mr-2 h-4 w-4 animate-spin" />
            )}
            Reset password
          </Button>
        </form>
        <FormMessage className="text-center" />
      </Form>
    </div>
  )
}
