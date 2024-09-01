'use client'

import * as React from 'react'
import { useRouter } from 'next/navigation'
import { zodResolver } from '@hookform/resolvers/zod'
import { isEmpty } from 'lodash-es'
import { useForm } from 'react-hook-form'
import { toast } from 'sonner'
import { useClient, useQuery } from 'urql'
import * as z from 'zod'

import { graphql } from '@/lib/gql/generates'
import { LicenseType, OAuthProvider } from '@/lib/gql/generates/graphql'
import { useMutation } from '@/lib/tabby/gql'
import { cn } from '@/lib/utils'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger
} from '@/components/ui/alert-dialog'
import { Button, buttonVariants } from '@/components/ui/button'
import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage
} from '@/components/ui/form'
import {
  IconGitHub,
  IconGitLab,
  IconGoogle,
  IconSpinner
} from '@/components/ui/icons'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group'
import { CopyButton } from '@/components/copy-button'
import { LicenseGuard } from '@/components/license-guard'

import { oauthCredential } from './oauth-credential-list'

export const updateOauthCredentialMutation = graphql(/* GraphQL */ `
  mutation updateOauthCredential($input: UpdateOAuthCredentialInput!) {
    updateOauthCredential(input: $input)
  }
`)

export const deleteOauthCredentialMutation = graphql(/* GraphQL */ `
  mutation deleteOauthCredential($provider: OAuthProvider!) {
    deleteOauthCredential(provider: $provider)
  }
`)

const oauthCallbackUrl = graphql(/* GraphQL */ `
  query OAuthCallbackUrl($provider: OAuthProvider!) {
    oauthCallbackUrl(provider: $provider)
  }
`)

const formSchema = z.object({
  clientId: z.string(),
  clientSecret: z.string().optional(),
  provider: z.nativeEnum(OAuthProvider)
})

export type OAuthCredentialFormValues = z.infer<typeof formSchema>

interface OAuthCredentialFormProps
  extends React.HTMLAttributes<HTMLDivElement> {
  isNew?: boolean
  provider: OAuthProvider
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

  const [deleteAlertVisible, setDeleteAlertVisible] = React.useState(false)
  const [isDeleting, setIsDeleting] = React.useState(false)

  const form = useForm<OAuthCredentialFormValues>({
    resolver: zodResolver(formSchema),
    defaultValues: formatedDefaultValues
  })
  const providerValue = form.watch('provider')
  const isDirty = !isEmpty(form.formState.dirtyFields)

  const { isSubmitting } = form.formState

  const navigateToSSOSettings = () => {
    router.replace('/settings/sso')
  }

  const updateOauthCredential = useMutation(updateOauthCredentialMutation, {
    onCompleted(values) {
      if (values?.updateOauthCredential) {
        onSuccess?.(form.getValues())
      }
    },
    form
  })

  const deleteOAuthCredential = useMutation(deleteOauthCredentialMutation)

  const onSubmit = async (values: OAuthCredentialFormValues) => {
    if (isNew) {
      const hasExistingProvider = await client
        .query(oauthCredential, { provider: values.provider })
        .then(res => !!res?.data?.oauthCredential)
      if (hasExistingProvider) {
        form.setError('provider', {
          message: 'Provider already exists. Please choose another one'
        })
        return
      }
    }

    updateOauthCredential({ input: values })
  }

  const onDelete: React.MouseEventHandler<HTMLButtonElement> = e => {
    e.preventDefault()
    setIsDeleting(true)
    deleteOAuthCredential({ provider: providerValue }).then(res => {
      if (res?.data?.deleteOauthCredential) {
        navigateToSSOSettings()
      } else {
        setIsDeleting(false)
        if (res?.error) {
          toast.error(res?.error?.message)
        }
      }
    })
  }

  const [{ data: oauthRedirectUrlData }] = useQuery({
    query: oauthCallbackUrl,
    variables: { provider: providerValue }
  })

  const accessTokenPlaceholder = React.useMemo(() => {
    if (!isNew) return new Array(36).fill('*').join('')

    return 'e.g. e363c08d7e9ca4e66e723a53f38a21f6a54c1b83'
  }, [isNew])

  return (
    <Form {...form}>
      <div className={cn('grid gap-2', className)} {...props}>
        <form className="grid gap-6" onSubmit={form.handleSubmit(onSubmit)}>
          <SubTitle className="mt-2">Basic information</SubTitle>
          <FormItem>
            <Label>Type</Label>
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
                    className="flex gap-8"
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
                      <Label
                        className="flex cursor-pointer items-center gap-2"
                        htmlFor="r_github"
                      >
                        <IconGitHub className="h-5 w-5" />
                        GitHub
                      </Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem
                        value={OAuthProvider.Google}
                        id="r_google"
                        disabled={!isNew}
                      />
                      <Label
                        className="flex cursor-pointer items-center gap-2"
                        htmlFor="r_google"
                      >
                        <IconGoogle className="h-5 w-5" />
                        Google
                      </Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem
                        value={OAuthProvider.Gitlab}
                        id="r_gitlab"
                        disabled={!isNew}
                      />
                      <Label
                        className="flex cursor-pointer items-center gap-2"
                        htmlFor="r_gitlab"
                      >
                        <IconGitLab className="h-5 w-5" />
                        GitLab
                      </Label>
                    </div>
                  </RadioGroup>
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />

          {oauthRedirectUrlData && (
            <FormItem className="mt-4">
              <div className="flex flex-col gap-2 rounded-lg border px-3 py-2">
                <div className="text-sm text-muted-foreground">
                  Create your OAuth2 application with the following information
                </div>
                <div className="flex items-center justify-between">
                  <div className="text-sm font-medium">
                    Authorization callback URL
                  </div>
                  <span className="flex items-center text-sm">
                    {oauthRedirectUrlData.oauthCallbackUrl}
                    <CopyButton
                      type="button"
                      value={oauthRedirectUrlData.oauthCallbackUrl!}
                    />
                  </span>
                </div>
              </div>
            </FormItem>
          )}

          <div>
            <SubTitle>OAuth provider information</SubTitle>
            <FormDescription>
              The information is provided by your identity provider.
            </FormDescription>
          </div>
          <FormField
            control={form.control}
            name="clientId"
            render={({ field }) => (
              <FormItem>
                <FormLabel required>Client ID</FormLabel>
                <FormControl>
                  <Input
                    placeholder="e.g. ae1542c44b154c10c859"
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
                <FormLabel required={isNew}>Client Secret</FormLabel>
                <FormControl>
                  <Input
                    className={cn({
                      'placeholder:translate-y-[10%] !placeholder-foreground':
                        !isNew
                    })}
                    placeholder={accessTokenPlaceholder}
                    autoCapitalize="none"
                    autoComplete="off"
                    autoCorrect="off"
                    type="password"
                    {...field}
                  />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
          <div className="flex justify-end gap-4">
            <Button
              type="button"
              variant="ghost"
              onClick={navigateToSSOSettings}
            >
              Cancel
            </Button>
            {!isNew && (
              <AlertDialog
                open={deleteAlertVisible}
                onOpenChange={setDeleteAlertVisible}
              >
                <AlertDialogTrigger asChild>
                  <Button variant="hover-destructive">Delete</Button>
                </AlertDialogTrigger>
                <AlertDialogContent>
                  <AlertDialogHeader>
                    <AlertDialogTitle>
                      Are you absolutely sure?
                    </AlertDialogTitle>
                    <AlertDialogDescription>
                      This action cannot be undone. It will permanently delete
                      the current credential.
                    </AlertDialogDescription>
                  </AlertDialogHeader>
                  <AlertDialogFooter>
                    <AlertDialogCancel>Cancel</AlertDialogCancel>
                    <AlertDialogAction
                      className={buttonVariants({ variant: 'destructive' })}
                      onClick={onDelete}
                    >
                      {isDeleting && (
                        <IconSpinner className="mr-2 h-4 w-4 animate-spin" />
                      )}
                      Yes, delete it
                    </AlertDialogAction>
                  </AlertDialogFooter>
                </AlertDialogContent>
              </AlertDialog>
            )}
            <LicenseGuard licenses={[LicenseType.Enterprise]}>
              {({ hasValidLicense }) => (
                <Button
                  type="submit"
                  disabled={
                    !hasValidLicense || isSubmitting || (!isNew && !isDirty)
                  }
                >
                  {isSubmitting && (
                    <IconSpinner className="mr-2 h-4 w-4 animate-spin" />
                  )}
                  {isNew ? 'Create' : 'Update'}
                </Button>
              )}
            </LicenseGuard>
          </div>
        </form>
        <FormMessage className="text-center" />
      </div>
    </Form>
  )
}

function SubTitle({
  className,
  ...rest
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={cn('mt-4 text-xl font-semibold', className)} {...rest} />
  )
}
