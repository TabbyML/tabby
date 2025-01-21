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
import { LdapEncryptionKind, LicenseType } from '@/lib/gql/generates/graphql'
import { useMutation } from '@/lib/tabby/gql'
import { ldapCredentialQuery } from '@/lib/tabby/query'
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
import { Checkbox } from '@/components/ui/checkbox'
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
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from '@/components/ui/select'
import { Separator } from '@/components/ui/separator'
import { LicenseGuard } from '@/components/license-guard'

import { SubTitle } from './form-sub-title'

const testLdapConnectionMutation = graphql(/* GraphQL */ `
  mutation testLdapConnection($input: UpdateLdapCredentialInput!) {
    testLdapConnection(input: $input)
  }
`)

const updateLdapCredentialMutation = graphql(/* GraphQL */ `
  mutation updateLdapCredential($input: UpdateLdapCredentialInput!) {
    updateLdapCredential(input: $input)
  }
`)

const deleteLdapCredentialMutation = graphql(/* GraphQL */ `
  mutation deleteLdapCredential {
    deleteLdapCredential
  }
`)

const formSchema = z.object({
  host: z.string(),
  port: z.coerce.number({
    required_error: 'Required',
    invalid_type_error: 'Invalid port'
  }),
  bindDn: z.string(),
  bindPassword: z.string().optional(),
  baseDn: z.string(),
  userFilter: z.string(),
  encryption: z.nativeEnum(LdapEncryptionKind),
  skipTlsVerify: z.boolean(),
  emailAttribute: z.string(),
  nameAttribute: z.string().optional()
})

export type LDAPFormValues = z.infer<typeof formSchema>

interface LDAPFormProps extends React.HTMLAttributes<HTMLDivElement> {
  isNew?: boolean
  defaultValues?: Partial<LDAPFormValues> | undefined
  onSuccess?: (formValues: LDAPFormValues) => void
  existed?: boolean
}

const providerExistedError =
  'LDAP provider already exists and cannot be created again.'

export function LDAPCredentialForm({
  className,
  isNew,
  defaultValues,
  onSuccess,
  existed,
  ...props
}: LDAPFormProps) {
  const router = useRouter()
  const client = useClient()
  const formRef = React.useRef<HTMLFormElement>(null)
  const [isTesting, setIsTesting] = React.useState(false)
  const formatedDefaultValues = React.useMemo(() => {
    return {
      ...(defaultValues || {})
    }
  }, [])

  const [deleteAlertVisible, setDeleteAlertVisible] = React.useState(false)
  const [isDeleting, setIsDeleting] = React.useState(false)

  const form = useForm<LDAPFormValues>({
    resolver: zodResolver(formSchema),
    defaultValues: formatedDefaultValues
  })
  const isDirty = !isEmpty(form.formState.dirtyFields)
  const isValid = form.formState.isValid
  const { isSubmitting } = form.formState

  const navigateToSSOSettings = () => {
    router.replace('/settings/sso')
  }

  const updateOauthCredential = useMutation(updateLdapCredentialMutation, {
    onCompleted(values) {
      if (values?.updateLdapCredential) {
        onSuccess?.(form.getValues())
      }
    },
    form
  })

  const testLdapConnection = useMutation(testLdapConnectionMutation, {
    onError(err) {
      toast.error(err.message)
    },
    onCompleted(data) {
      if (data?.testLdapConnection) {
        toast.success('LDAP connection test success.', {
          className: 'mb-10'
        })
      } else {
        toast.error('LDAP connection test failed.')
      }
    }
  })

  const deleteLdapCredential = useMutation(deleteLdapCredentialMutation)

  const onSubmit = async (values: LDAPFormValues) => {
    if (isNew) {
      const hasExistingProvider = await client
        .query(ldapCredentialQuery, {})
        .then(res => !!res?.data?.ldapCredential)
      if (hasExistingProvider) {
        form.setError('root', {
          message: providerExistedError
        })
        return
      }
    }

    return updateOauthCredential({ input: values })
  }

  const onDelete: React.MouseEventHandler<HTMLButtonElement> = e => {
    e.preventDefault()
    setIsDeleting(true)
    deleteLdapCredential().then(res => {
      if (res?.data?.deleteLdapCredential) {
        navigateToSSOSettings()
      } else {
        setIsDeleting(false)
        if (res?.error) {
          toast.error(res?.error?.message)
        }
      }
    })
  }

  const onTestLdapCredential = () => {
    if (!formRef.current) return
    form.trigger().then(isValid => {
      if (!isValid) return

      setIsTesting(true)

      return testLdapConnection({
        input: formSchema.parse(form.getValues())
      }).finally(() => {
        setIsTesting(false)
      })
    })
  }

  const passwordPlaceholder = React.useMemo(() => {
    if (!isNew) return new Array(36).fill('*').join('')

    return undefined
  }, [isNew])

  return (
    <Form {...form}>
      <div className={cn('grid gap-2', className)} {...props}>
        {existed && (
          <div className="mt-2 text-sm font-medium text-destructive">
            {providerExistedError}
          </div>
        )}
        <form
          className="mt-6 grid gap-4"
          onSubmit={form.handleSubmit(onSubmit)}
          ref={formRef}
        >
          <div>
            <SubTitle>LDAP provider information</SubTitle>
            <FormDescription>
              The information is provided by your identity provider.
            </FormDescription>
          </div>
          <div className="flex flex-col gap-6 lg:flex-row">
            <FormField
              control={form.control}
              name="host"
              render={({ field }) => (
                <FormItem>
                  <FormLabel required>Host</FormLabel>
                  <FormControl>
                    <Input
                      placeholder="e.g. ldap.example.com"
                      autoCapitalize="none"
                      autoComplete="off"
                      autoCorrect="off"
                      className="w-80 min-w-max"
                      {...field}
                    />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="port"
              render={({ field }) => (
                <FormItem>
                  <FormLabel required>Port</FormLabel>
                  <FormControl>
                    <Input
                      placeholder="e.g. 3890"
                      autoCapitalize="none"
                      autoComplete="off"
                      autoCorrect="off"
                      className="w-80 min-w-max"
                      {...field}
                    />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
          </div>
          <div className="flex flex-col gap-6 lg:flex-row">
            <FormField
              control={form.control}
              name="bindDn"
              render={({ field }) => (
                <FormItem>
                  <FormLabel required>Bind DN</FormLabel>
                  <FormControl>
                    <Input
                      placeholder="e.g. uid=system,ou=Users,dc=example,dc=com"
                      autoCapitalize="none"
                      autoComplete="off"
                      autoCorrect="off"
                      className="w-80 min-w-max"
                      {...field}
                    />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="bindPassword"
              render={({ field }) => (
                <FormItem>
                  <FormLabel required={isNew}>Bind Password</FormLabel>
                  <FormControl>
                    <Input
                      className={cn('w-80 min-w-max', {
                        'placeholder:translate-y-[10%] !placeholder-foreground/50':
                          !isNew
                      })}
                      placeholder={passwordPlaceholder}
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
          </div>
          <FormField
            control={form.control}
            name="baseDn"
            render={({ field }) => (
              <FormItem>
                <FormLabel required>Base DN</FormLabel>
                <FormControl>
                  <Input
                    placeholder="e.g. ou=Users,dc=example,dc=com"
                    autoCapitalize="none"
                    autoComplete="off"
                    autoCorrect="off"
                    className="w-80 min-w-max"
                    {...field}
                  />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
          <FormField
            control={form.control}
            name="userFilter"
            render={({ field }) => (
              <FormItem>
                <FormLabel required>User Filter</FormLabel>
                <FormControl>
                  <Input
                    placeholder="e.g. (uid=%s)"
                    autoCapitalize="none"
                    autoComplete="off"
                    autoCorrect="off"
                    className="w-80 min-w-max"
                    {...field}
                  />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
          <FormField
            control={form.control}
            name="encryption"
            render={({ field }) => (
              <FormItem>
                <FormLabel required>Encryption</FormLabel>
                <Select
                  onValueChange={field.onChange}
                  defaultValue={field.value}
                  name={field.name}
                >
                  <FormControl>
                    <SelectTrigger className="w-80 min-w-max">
                      <SelectValue placeholder="Select an encryption" />
                    </SelectTrigger>
                  </FormControl>
                  <SelectContent>
                    <SelectItem value={LdapEncryptionKind.None}>
                      NONE
                    </SelectItem>
                    <SelectItem value={LdapEncryptionKind.StartTls}>
                      STARTTLS
                    </SelectItem>
                    <SelectItem value={LdapEncryptionKind.Ldaps}>
                      LDAPS
                    </SelectItem>
                  </SelectContent>
                </Select>
                <FormMessage />
              </FormItem>
            )}
          />
          <FormField
            control={form.control}
            name="skipTlsVerify"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Connection security</FormLabel>
                <div className="flex items-center gap-1">
                  <FormControl>
                    <Checkbox
                      checked={field.value}
                      onCheckedChange={field.onChange}
                    />
                  </FormControl>
                  <FormLabel className="cursor-pointer">
                    Skip TLS Verify
                  </FormLabel>
                </div>
                <FormMessage />
              </FormItem>
            )}
          />
          <div className="mt-4">
            <SubTitle>User information mapping</SubTitle>
            <FormDescription>
              Maps the field names from user info API to the Tabby user.
            </FormDescription>
          </div>
          <FormField
            control={form.control}
            name="emailAttribute"
            render={({ field }) => (
              <FormItem>
                <FormLabel required>Email</FormLabel>
                <FormControl>
                  <Input
                    autoCapitalize="none"
                    autoComplete="off"
                    autoCorrect="off"
                    className="w-80 min-w-max"
                    placeholder="e.g. email"
                    {...field}
                  />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
          <FormField
            control={form.control}
            name="nameAttribute"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Name</FormLabel>
                <FormControl>
                  <Input
                    autoCapitalize="none"
                    autoComplete="off"
                    autoCorrect="off"
                    className="w-80 min-w-max"
                    placeholder="e.g. name"
                    {...field}
                  />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
          <Separator className="my-2" />
          <div className="flex flex-col gap-4 sm:flex-row sm:justify-between">
            <Button
              onClick={onTestLdapCredential}
              type="button"
              variant="outline"
              disabled={(isNew && !isValid) || isTesting}
            >
              Test Connection
              {isTesting && (
                <IconSpinner className="mr-2 h-4 w-4 animate-spin" />
              )}
            </Button>
            <div className="flex items-center justify-end gap-4 sm:justify-start">
              <Button
                type="button"
                variant="ghost"
                onClick={navigateToSSOSettings}
              >
                Back
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
          </div>
        </form>
        <FormMessage className="text-center" />
      </div>
    </Form>
  )
}
