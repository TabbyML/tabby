'use client'

import * as React from 'react'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { zodResolver } from '@hookform/resolvers/zod'
import { useForm } from 'react-hook-form'
import * as z from 'zod'

import { PLACEHOLDER_EMAIL_FORM } from '@/lib/constants'
import { graphql } from '@/lib/gql/generates'
import { useIsEmailConfigured } from '@/lib/hooks/use-server-info'
import { useSession, useSignIn } from '@/lib/tabby/auth'
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

export const tokenAuth = graphql(/* GraphQL */ `
  mutation tokenAuth($email: String!, $password: String!) {
    tokenAuth(email: $email, password: $password) {
      accessToken
      refreshToken
    }
  }
`)

const formSchema = z.object({
  email: z.string().email('Invalid email address'),
  password: z.string()
})

interface UserAuthFormProps extends React.HTMLAttributes<HTMLDivElement> {
  invitationCode?: string
}

export default function UserSignInForm({
  className,
  invitationCode,
  ...props
}: UserAuthFormProps) {
  const isEmailConfigured = useIsEmailConfigured()
  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema)
  })

  const router = useRouter()
  const { status } = useSession()
  React.useEffect(() => {
    if (status === 'authenticated') {
      router.replace('/')
    }
  }, [status])

  const signIn = useSignIn()
  const { isSubmitting } = form.formState
  const onSubmit = useMutation(tokenAuth, {
    onCompleted(values) {
      signIn(values.tokenAuth)
    },
    form
  })

  return (
    <div className={cn('grid', className)} {...props}>
      <Form {...form}>
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
          <FormField
            control={form.control}
            name="password"
            render={({ field }) => (
              <FormItem>
                <div className="flex items-center justify-between">
                  <FormLabel>Password</FormLabel>
                  {!!isEmailConfigured && (
                    <div className="cursor-pointer text-right text-sm text-primary hover:underline">
                      <Link href="/auth/signin?mode=reset">
                        Forgot password?
                      </Link>
                    </div>
                  )}
                </div>
                <FormControl>
                  <Input type="password" {...field} />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
          <Button type="submit" className="mt-1" disabled={isSubmitting}>
            {isSubmitting && (
              <IconSpinner className="mr-2 h-4 w-4 animate-spin" />
            )}
            Login
          </Button>
        </form>
        <FormMessage className="text-center" />
      </Form>
    </div>
  )
}
