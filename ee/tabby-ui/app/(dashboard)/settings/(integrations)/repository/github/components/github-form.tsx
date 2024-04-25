'use client'

import * as React from 'react'
import Link from 'next/link'
import { zodResolver } from '@hookform/resolvers/zod'
import { useForm, UseFormReturn } from 'react-hook-form'
import * as z from 'zod'

import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage
} from '@/components/ui/form'
import { IconExternalLink } from '@/components/ui/icons'
import { Input } from '@/components/ui/input'

export const createGithubProviderSchema = z.object({
  provider: z.string(),
  displayName: z
    .string()
    .trim()
    .regex(
      /^[\w-]+$/,
      'Display name must contain only alphanumeric characters, underscores, and hyphens'
    ),
  accessToken: z.string()
})

export const updateGithubProviderSchema = createGithubProviderSchema.extend({
  provider: createGithubProviderSchema.shape.provider.optional()
})

export type CreateGithubProviderFormValues = z.infer<
  typeof createGithubProviderSchema
>
export type UpdateGithubProviderFormValues = z.infer<
  typeof updateGithubProviderSchema
>

interface GithubProviderFormProps {
  isNew?: boolean
  defaultValues?: Partial<z.infer<typeof createGithubProviderSchema>>
  footer: React.ReactNode
  onSubmit: (values: any) => Promise<any>
}

export const GithubProviderForm = React.forwardRef<
  {
    form: UseFormReturn<
      CreateGithubProviderFormValues | UpdateGithubProviderFormValues
    >
  },
  GithubProviderFormProps
>(({ isNew, defaultValues, footer, onSubmit }, ref) => {
  const formSchema = isNew
    ? createGithubProviderSchema
    : updateGithubProviderSchema
  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues
  })

  React.useImperativeHandle(
    ref,
    () => {
      return {
        form
      }
    },
    [form]
  )

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
                    placeholder="e.g. GitHub"
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
                <FormLabel required={isNew}>Personal Access Token</FormLabel>
                <FormDescription>
                  <div>
                    Create a dedicated service user and generate a{' '}
                    <ExternalLink href="https://github.com/settings/personal-access-tokens/new">
                      fine-grained personal access
                    </ExternalLink>{' '}
                    token with the member role for the organization or all
                    projects to be managed.
                  </div>
                  <div className="my-2 ml-4">• Contents (Read-only)</div>
                </FormDescription>
                <FormControl>
                  <Input
                    placeholder="e.g. github_pat_1ABCD1234ABCD1234ABCD1234ABCD1234ABCD1234"
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
          {footer}
        </form>
      </div>
    </Form>
  )
})

GithubProviderForm.displayName = 'GithubProviderForm'

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
