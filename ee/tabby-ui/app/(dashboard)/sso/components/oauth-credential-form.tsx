'use client'

import * as React from 'react'
import { zodResolver } from '@hookform/resolvers/zod'
import { useForm } from 'react-hook-form'
import { toast } from 'sonner'
import * as z from 'zod'

import { graphql } from '@/lib/gql/generates'
import { OAuthProvider } from '@/lib/gql/generates/graphql'
import { useMutation } from '@/lib/tabby/gql'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage
} from '@/components/ui/form'
import { IconSpinner } from '@/components/ui/icons'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group'
import { CopyButton } from '@/components/copy-button'

export const updateOauthCredentialMutation = graphql(/* GraphQL */ `
  mutation updateOauthCredential(
    $provider: OAuthProvider!
    $clientId: String!
    $clientSecret: String!
    $redirectUri: String
  ) {
    updateOauthCredential(
      provider: $provider
      clientId: $clientId
      clientSecret: $clientSecret
      redirectUri: $redirectUri
    )
  }
`)

const formSchema = z.object({
  clientId: z.string(),
  clientSecret: z.string(),
  provider: z.nativeEnum(OAuthProvider)
})

export type OAuthCredentialFormValues = z.infer<typeof formSchema>

interface OAuthCredentialFormProps
  extends React.HTMLAttributes<HTMLDivElement> {
  provider?: OAuthProvider
  defaultValues?: Partial<OAuthCredentialFormValues> | undefined
  onSuccess?: (formValues: OAuthCredentialFormValues) => void
}

export default function OAuthCredentialForm({
  className,
  provider,
  defaultValues,
  onSuccess,
  ...props
}: OAuthCredentialFormProps) {
  const formatedDefaultValues = React.useMemo(() => {
    return {
      ...(defaultValues || {}),
      provider
    }
  }, [])
  const isNew = !defaultValues

  const form = useForm<OAuthCredentialFormValues>({
    resolver: zodResolver(formSchema),
    defaultValues: formatedDefaultValues
  })
  const providerValue = form.watch('provider')

  const { isSubmitting } = form.formState

  const oauthRedirectUri = React.useMemo(() => {
    if (!providerValue) return ''

    let origin = window.location.origin
    if (process.env.NODE_ENV !== 'production') {
      origin = `${process.env.NEXT_PUBLIC_TABBY_SERVER_URL}` ?? origin
    }

    return `${origin}/oauth/callback/${providerValue.toLowerCase()}`
  }, [providerValue])

  const updateOauthCredential = useMutation(updateOauthCredentialMutation, {
    onCompleted(values) {
      if (values?.updateOauthCredential) {
        toast.success(`success`)
        onSuccess?.(form.getValues())
      }
    },
    form
  })

  const onSubmit = (values: OAuthCredentialFormValues) => {
    // add redirect uri automatically
    updateOauthCredential({ ...values, redirectUri: oauthRedirectUri })
  }

  return (
    <div className={cn('grid gap-6', className)} {...props}>
      <Form {...form}>
        <form className="grid gap-4" onSubmit={form.handleSubmit(onSubmit)}>
          <FormItem>
            <FormLabel>Type</FormLabel>
            <RadioGroup defaultValue="oauth">
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="oauth" id="type_oauth" />
                <Label className="cursor-pointer" htmlFor="type_oauth">
                  OAuth 2.0
                </Label>
              </div>
            </RadioGroup>
          </FormItem>
          <FormField
            control={form.control}
            name="provider"
            render={({ field: { onChange, ...rest } }) => (
              <FormItem>
                <FormLabel>Provider</FormLabel>
                <FormControl>
                  <RadioGroup
                    className="flex gap-6"
                    orientation="horizontal"
                    onValueChange={onChange}
                    {...rest}
                  >
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem
                        value={OAuthProvider.Github}
                        id="r_github"
                        disabled={!isNew}
                      />
                      <Label className="cursor-pointer" htmlFor="r_github">
                        Github
                      </Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem
                        value={OAuthProvider.Google}
                        id="r_google"
                        disabled={!isNew}
                      />
                      <Label className="cursor-pointer" htmlFor="r_google">
                        Google
                      </Label>
                    </div>
                  </RadioGroup>
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
          <FormField
            control={form.control}
            name="clientId"
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
          <FormItem>
            <FormDescription>
              Use this to create your oauth application
            </FormDescription>
            <FormLabel>Authorization callback URL</FormLabel>
            {oauthRedirectUri ? (
              <div>
                <div
                  className="inline-flex items-center gap-4 rounded-lg border p-2"
                  onClick={e => e.stopPropagation()}
                >
                  <span className="text-sm">{oauthRedirectUri}</span>
                  {!!providerValue && (
                    <CopyButton type="button" value={oauthRedirectUri} />
                  )}
                </div>
              </div>
            ) : (
              <div className="text-sm text-muted-foreground">
                Please select a provider
              </div>
            )}
          </FormItem>
          <Button type="submit" className="mt-1" disabled={isSubmitting}>
            {isSubmitting && (
              <IconSpinner className="mr-2 h-4 w-4 animate-spin" />
            )}
            {isNew ? 'Submit' : 'Update'}
          </Button>
        </form>
        <FormMessage className="text-center" />
      </Form>
    </div>
  )
}
