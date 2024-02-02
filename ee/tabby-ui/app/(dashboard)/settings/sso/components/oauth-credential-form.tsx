'use client'

import * as React from 'react'
import { useRouter } from 'next/navigation'
import { zodResolver } from '@hookform/resolvers/zod'
import { isEmpty } from 'lodash-es'
import { useForm } from 'react-hook-form'
import { toast } from 'sonner'
import { useClient } from 'urql'
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

import { oauthCredential } from './oauth-credential-list'

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
  isNew?: boolean
  provider?: OAuthProvider
  defaultValues?: Partial<OAuthCredentialFormValues> | undefined
  onSuccess?: (formValues: OAuthCredentialFormValues) => void
}

export default function OAuthCredentialForm({
  className,
  isNew,
  provider,
  defaultValues,
  onSuccess,
  ...props
}: OAuthCredentialFormProps) {
  const router = useRouter()
  const client = useClient()
  const formatedDefaultValues = React.useMemo(() => {
    return {
      ...(defaultValues || {}),
      provider
    }
  }, [])

  const [oauthRedirectUri, setOAuthRedirectUri] = React.useState<string>('')

  const form = useForm<OAuthCredentialFormValues>({
    resolver: zodResolver(formSchema),
    defaultValues: formatedDefaultValues
  })
  const providerValue = form.watch('provider')
  const isDirty = !isEmpty(form.formState.dirtyFields)

  const { isSubmitting } = form.formState

  React.useEffect(() => {
    if (providerValue) {
      // FIXME  use public origin
      let origin = window.location.origin
      if (process.env.NODE_ENV !== 'production') {
        origin = `${process.env.NEXT_PUBLIC_TABBY_SERVER_URL}` ?? origin
      }
      const uri = `${origin}/oauth/callback/${providerValue.toLowerCase()}`
      setOAuthRedirectUri(uri)
    }
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

  const onSubmit = async (values: OAuthCredentialFormValues) => {
    client.query(oauthCredential, { provider: values.provider }).then(res => {
      if (res?.data?.oauthCredential) {
        // if aleardy have, set Form Error
        form.setError('provider', {
          message: 'Provider already exists. Please choose another one'
        })
        return
      }

      // add redirect uri automatically
      updateOauthCredential({ ...values, redirectUri: oauthRedirectUri })
    })
  }

  return (
    <div className={cn('grid gap-6', className)} {...props}>
      <Form {...form}>
        <form className="grid gap-4" onSubmit={form.handleSubmit(onSubmit)}>
          <SubTitle className="mt-2">Basic information</SubTitle>
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

          <div>
            <SubTitle>Identity provider information</SubTitle>
            <FormDescription>
              The information is provided by your identity provider.
            </FormDescription>
          </div>
          <FormItem>
            <div className="rounded-lg border px-3 py-2 flex flex-col gap-2">
              <div className="text-muted-foreground text-sm">
                Create your SSO application with the following information
              </div>
              <div className="flex items-center gap-4">
                <div className="font-medium text-sm">
                  Authorization redirect URL
                </div>
                <span className="text-sm">{oauthRedirectUri}</span>
                {!!providerValue && (
                  <CopyButton type="button" value={oauthRedirectUri} />
                )}
              </div>
            </div>
          </FormItem>
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
                    autoComplete="off"
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
                  <Input
                    {...field}
                    placeholder={isNew ? undefined : 'sensitive - write only'}
                    autoCapitalize="none"
                    autoComplete="off"
                    autoCorrect="off"
                  />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
          <div className="flex gap-4 justify-end mt-1">
            {isNew ? (
              <Button
                type="button"
                variant="ghost"
                onClick={e => router.replace('/settings/sso')}
              >
                Cancel
              </Button>
            ) : (
              <Button
                type="button"
                variant="ghost"
                onClick={e => form.reset()}
                disabled={isSubmitting || !isDirty}
              >
                Discard changes
              </Button>
            )}
            <Button
              type="submit"
              disabled={isSubmitting || (!isNew && !isDirty)}
            >
              {isSubmitting && (
                <IconSpinner className="mr-2 h-4 w-4 animate-spin" />
              )}
              {isNew ? 'Create' : 'Update'}
            </Button>
          </div>
        </form>
        <FormMessage className="text-center" />
      </Form>
    </div>
  )
}

function SubTitle({
  className,
  ...rest
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={cn('font-semibold text-xl mt-4', className)} {...rest} />
  )
}
