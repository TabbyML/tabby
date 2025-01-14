'use client'

import * as React from 'react'
import { zodResolver } from '@hookform/resolvers/zod'
import { useForm } from 'react-hook-form'
import * as z from 'zod'

import { graphql } from '@/lib/gql/generates'
import { useSignIn } from '@/lib/tabby/auth'
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

const tokenAuthLdap = graphql(/* GraphQL */ `
  mutation tokenAuthLdap($userId: String!, $password: String!) {
    tokenAuthLdap(userId: $userId, password: $password) {
      accessToken
      refreshToken
    }
  }
`)

const formSchema = z.object({
  userId: z.string().trim(),
  password: z.string().trim()
})

interface LdapAuthFormProps extends React.HTMLAttributes<HTMLDivElement> {
  invitationCode?: string
}

export default function LdapSignInForm({
  className,
  invitationCode,
  ...props
}: LdapAuthFormProps) {
  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema)
  })
  const formRef = React.useRef<HTMLFormElement | null>(null)

  const signIn = useSignIn()
  const { isSubmitting } = form.formState
  const onSubmit = useMutation(tokenAuthLdap, {
    onCompleted(values) {
      signIn(values.tokenAuthLdap)
    },
    form
  })

  return (
    <Form {...form}>
      <div className={cn('grid gap-2', className)} {...props}>
        <form
          ref={formRef}
          className="grid gap-4"
          onSubmit={form.handleSubmit(onSubmit)}
        >
          <FormField
            control={form.control}
            name="userId"
            render={({ field }) => (
              <FormItem>
                <FormLabel className="leading-5">Username</FormLabel>
                <FormControl>
                  <Input
                    placeholder="name"
                    autoCapitalize="none"
                    autoCorrect="off"
                    {...field}
                  />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
          <FormField
            control={form.control}
            name="password"
            render={({ field }) => (
              <FormItem>
                <div className="flex items-center justify-between">
                  <FormLabel className="leading-5">Password</FormLabel>
                </div>
                <FormControl>
                  <Input type="password" {...field} />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
          <Button type="submit" className="mt-2" disabled={isSubmitting}>
            {isSubmitting && (
              <IconSpinner className="mr-2 h-4 w-4 animate-spin" />
            )}
            Sign In
          </Button>
        </form>
        <FormMessage className="text-center" />
      </div>
    </Form>
  )
}
