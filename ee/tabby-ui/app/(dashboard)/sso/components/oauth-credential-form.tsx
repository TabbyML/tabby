'use client'

import * as React from 'react'
import { useRouter } from 'next/navigation'
import { zodResolver } from '@hookform/resolvers/zod'
import { useForm } from 'react-hook-form'
import * as z from 'zod'

import { graphql } from '@/lib/gql/generates'
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
import { OAuthProvider } from '@/lib/gql/generates/graphql'

export const updateOauthCredentialMutation = graphql(/* GraphQL */ `
  mutation updateOauthCredential($provider: OAuthProvider!, $clientId: String!, $clientSecret: String!, $redirectUri: String) {
    updateOauthCredential(provider: $provider, clientId: $clientId, clientSecret: $clientSecret, redirectUri: $redirectUri)
  }
`)

const formSchema = z.object({
  clientId: z.string(),
  clientSecret: z.string(),
  redirectUri: z.string(),
})

interface OAuthCredentialFormProps extends React.HTMLAttributes<HTMLDivElement> {
  provider: OAuthProvider
}

export default function UserSignInForm({
  className,
  provider,
  ...props
}: OAuthCredentialFormProps) {
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
  const onSubmit = useMutation(updateOauthCredentialMutation, {
    onCompleted(values) {
      console.log(values.updateOauthCredential)
      
    },
    form
  })

  return (
    <div className={cn('grid gap-6', className)} {...props}>
      <Form {...form}>
        <form className="grid gap-2" onSubmit={form.handleSubmit(values => onSubmit({ provider, ...values }))}>
          <FormField
            control={form.control}
            name='clientId'
            render={({ field }) => (
              <FormItem>
                <FormLabel>Client ID</FormLabel>
                <FormControl>
                  <Input
                    placeholder=""
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
            name="clientSecret"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Client Secret</FormLabel>
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
