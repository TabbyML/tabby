'use client'

import * as React from 'react'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { zodResolver } from '@hookform/resolvers/zod'
import { isEmpty } from 'lodash-es'
import { DefaultValues, useForm, UseFormReturn } from 'react-hook-form'
import { toast } from 'sonner'
import * as z from 'zod'

import { RepositoryKind } from '@/lib/gql/generates/graphql'
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
import { IconExternalLink, IconSpinner } from '@/components/ui/icons'
import { Input } from '@/components/ui/input'

import { useRepositoryKind } from '../hooks/use-repository-kind'

export const createRepositoryProviderFormSchema = z.object({
  displayName: z.string().trim(),
  accessToken: z.string()
})

export const updateRepositoryProviderFormSchema =
  createRepositoryProviderFormSchema.extend({
    accessToken: z.string().optional()
  })

export type CreateRepositoryProviderFormValues = z.infer<
  typeof createRepositoryProviderFormSchema
>

export type UpdateRepositoryProviderFormValues = z.infer<
  typeof updateRepositoryProviderFormSchema
>

type FormValues<T extends boolean> = T extends true
  ? CreateRepositoryProviderFormValues
  : UpdateRepositoryProviderFormValues

interface GithubProviderFormProps<T extends boolean> {
  isNew: T
  form: UseFormReturn<UpdateRepositoryProviderFormValues>
  onSubmit: (values: any) => Promise<any>
  onDelete?: () => Promise<any>
  cancleable?: boolean
  deletable?: boolean
}

export function CommonProviderForm<T extends boolean>({
  isNew,
  form,
  onSubmit,
  onDelete,
  cancleable = true,
  deletable
}: GithubProviderFormProps<T>) {
  const kind = useRepositoryKind()
  const router = useRouter()

  const [deleteAlertVisible, setDeleteAlertVisible] = React.useState(false)
  const [isDeleting, setIsDeleting] = React.useState(false)

  const { isSubmitting, dirtyFields } = form.formState
  const isDirty = !isEmpty(dirtyFields)

  const handleDeleteRepositoryProvider: React.MouseEventHandler<
    HTMLButtonElement
  > = async e => {
    e.preventDefault()
    if (!onDelete) return

    setIsDeleting(true)
    try {
      await onDelete()
    } catch (error) {
      toast.error('Failed to delete GitHub repository provider')
    } finally {
      setIsDeleting(false)
    }
  }

  const displayNamePlaceholder = React.useMemo(() => {
    switch (kind) {
      case RepositoryKind.Github:
        return 'e.g. GitHub'
      case RepositoryKind.Gitlab:
        return 'e.g. GitLab'
      default:
        return ''
    }
  }, [kind])

  const accessTokenPlaceholder = React.useMemo(() => {
    if (!isNew) return new Array(36).fill('*').join('')
    switch (kind) {
      case RepositoryKind.Github:
        return 'e.g. github_pat_1ABCD1234ABCD1234ABCD1234ABCD1234ABCD1234'
      case RepositoryKind.Gitlab:
        return 'e.g. glpat_1ABCD1234ABCD1234ABCD1234ABCD1234'
      default:
        return ''
    }
  }, [kind, isNew])

  return (
    <Form {...form}>
      <div className="grid gap-2">
        <form className="grid gap-6" onSubmit={form.handleSubmit(onSubmit)}>
          <FormField
            control={form.control}
            name="displayName"
            render={({ field }) => (
              <FormItem>
                <FormLabel required>Display name</FormLabel>
                <FormDescription>
                  A display name to help identifying different providers.
                </FormDescription>
                <FormControl>
                  <Input
                    placeholder={displayNamePlaceholder}
                    autoCapitalize="none"
                    autoCorrect="off"
                    autoComplete="off"
                    {...field}
                  />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
          <FormField
            control={form.control}
            name="accessToken"
            render={({ field }) => (
              <FormItem>
                <FormLabel required>Personal Access Token</FormLabel>
                <FormDescription>
                  <AccessTokenDescription />
                </FormDescription>
                <FormControl>
                  <Input
                    placeholder={accessTokenPlaceholder}
                    className={cn({
                      'placeholder:translate-y-[10%] !placeholder-foreground':
                        !isNew
                    })}
                    autoCapitalize="none"
                    autoCorrect="off"
                    autoComplete="off"
                    {...field}
                  />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
          <div className="flex items-center justify-between">
            <div>
              <FormMessage />
            </div>
            <div className="flex gap-2">
              {cancleable && (
                <Button
                  type="button"
                  variant="ghost"
                  disabled={isSubmitting}
                  onClick={() => router.back()}
                >
                  Cancel
                </Button>
              )}
              {deletable && (
                <AlertDialog
                  open={deleteAlertVisible}
                  onOpenChange={setDeleteAlertVisible}
                >
                  <AlertDialogTrigger asChild>
                    <Button type="button" variant="hover-destructive">
                      Delete
                    </Button>
                  </AlertDialogTrigger>
                  <AlertDialogContent>
                    <AlertDialogHeader>
                      <AlertDialogTitle>
                        Are you absolutely sure?
                      </AlertDialogTitle>
                      <AlertDialogDescription>
                        This will delete the provider and remove any
                        repositories that have already been added to the
                        provider.
                      </AlertDialogDescription>
                    </AlertDialogHeader>
                    <AlertDialogFooter>
                      <AlertDialogCancel>Cancel</AlertDialogCancel>
                      <AlertDialogAction
                        className={buttonVariants({ variant: 'destructive' })}
                        onClick={handleDeleteRepositoryProvider}
                        disabled={isDeleting}
                      >
                        {isDeleting && <IconSpinner className="mr-2" />}
                        Yes, delete it
                      </AlertDialogAction>
                    </AlertDialogFooter>
                  </AlertDialogContent>
                </AlertDialog>
              )}
              <Button
                type="submit"
                disabled={isSubmitting || (!isNew && !isDirty)}
              >
                {isSubmitting && <IconSpinner className="mr-2" />}
                {isNew ? 'Create' : 'Update'}
              </Button>
            </div>
          </div>
        </form>
      </div>
    </Form>
  )
}

export function useRepositoryProviderForm<T extends boolean>(
  isNew: T,
  defaultValues?: Partial<FormValues<T>>
): UseFormReturn<FormValues<T>> {
  const schema = isNew
    ? createRepositoryProviderFormSchema
    : updateRepositoryProviderFormSchema
  return useForm<FormValues<T>>({
    resolver: zodResolver(schema),
    defaultValues: defaultValues as DefaultValues<FormValues<T>>
  })
}

function AccessTokenDescription() {
  const kind = useRepositoryKind()
  if (kind === RepositoryKind.Github) {
    return (
      <>
        <div>
          Create a dedicated service user and generate a{' '}
          <ExternalLink href="https://github.com/settings/personal-access-tokens/new">
            fine-grained personal access
          </ExternalLink>{' '}
          token with the member role for the organization or all projects to be
          managed.
        </div>
        <div className="my-2 ml-3">• Contents (Read-only)</div>
      </>
    )
  }

  if (kind === RepositoryKind.Gitlab) {
    return (
      <>
        <div>
          Create a dedicated service user and generate a{' '}
          <ExternalLink href="https://gitlab.com/-/user_settings/personal_access_tokens">
            personal access token
          </ExternalLink>{' '}
          with the maintainer role and at least following permissions for the
          group or projects to be managed. You can generate a project access
          token for managing a single project, or generate a group access token
          to manage all projects within the group.
        </div>
        <div className="my-2 ml-3">• api</div>
        <div className="my-2 ml-3">• read repository</div>
      </>
    )
  }

  return null
}

function ExternalLink({
  href,
  children
}: {
  href: string
  children: React.ReactNode
}) {
  return (
    <Link
      className="inline-flex cursor-pointer flex-row items-center underline"
      href={href}
      target="_blank"
    >
      {children}
      <IconExternalLink className="ml-1" />
    </Link>
  )
}
