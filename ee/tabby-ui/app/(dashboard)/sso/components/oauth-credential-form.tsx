'use client'

import * as React from 'react'
import { zodResolver } from '@hookform/resolvers/zod'
import { useForm } from 'react-hook-form'
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
import { OAuthCredential, OAuthProvider } from '@/lib/gql/generates/graphql'
import { Label } from "@/components/ui/label"
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group'
import { useRouter } from 'next/navigation'
import { find } from 'lodash-es'
import { PARAMS_TO_ENUM } from '../detail/[provider]/page'

export const updateOauthCredentialMutation = graphql(/* GraphQL */ `
  mutation updateOauthCredential($provider: OAuthProvider!, $clientId: String!, $clientSecret: String!, $redirectUri: String) {
    updateOauthCredential(provider: $provider, clientId: $clientId, clientSecret: $clientSecret, redirectUri: $redirectUri)
  }
`)

const formSchema = z.object({
  clientId: z.string(),
  clientSecret: z.string(),
  redirectUri: z.string(),
  provider: z.nativeEnum(OAuthProvider)
})

interface OAuthCredentialFormProps extends React.HTMLAttributes<HTMLDivElement> {
  provider?: OAuthProvider
  defaultValues?: OAuthCredential
}

export default function OAuthCredentialForm({
  className,
  provider,
  defaultValues,
  ...props
}: OAuthCredentialFormProps) {
  const router = useRouter()
  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    // defaultValues: defaultValues ?? undefined
  })

  const { isSubmitting } = form.formState
  const onSubmit = useMutation(updateOauthCredentialMutation, {
    onCompleted(values) {
      if (values?.updateOauthCredential) {
        const _provider = form.getValues()?.provider
        const pathSegment = find(PARAMS_TO_ENUM, p => p.enum === _provider)
        router.push(`/sso/detail/${pathSegment?.name}`)
      }
    },
    form
  })

  return (
    <div className={cn('grid gap-6', className)} {...props}>
      <Form {...form}>
        <form className="grid gap-4" onSubmit={form.handleSubmit(onSubmit)}>
          <FormField
            control={form.control}
            name='provider'
            render={({ field: { onChange, ...rest } }) => (
              <FormItem>
                <FormLabel>Provider</FormLabel>
                <FormControl>
                  <RadioGroup className='flex gap-6' orientation='horizontal' {...rest} onValueChange={onChange} >
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value={OAuthProvider.Github} id="r_github" />
                      <Label className='cursor-pointer' htmlFor="r_github">Github</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value={OAuthProvider.Google} id="r_google" />
                      <Label className='cursor-pointer' htmlFor="r_google">Google</Label>
                    </div>
                  </RadioGroup>
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
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
                  <Input {...field} />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
          <FormField
            control={form.control}
            name='redirectUri'
            render={({ field }) => (
              <FormItem>
                <FormLabel>Redirect URI</FormLabel>
                <FormControl>
                  <Input {...field} />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
          <Button type="submit" className="mt-1" disabled={isSubmitting}>
            {isSubmitting && (
              <IconSpinner className="mr-2 h-4 w-4 animate-spin" />
            )}
            Submit
          </Button>
        </form>
        <FormMessage className="text-center" />
      </Form>
    </div>
  )
}
