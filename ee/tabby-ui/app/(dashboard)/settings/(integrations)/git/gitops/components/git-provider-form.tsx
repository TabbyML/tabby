'use client'

import * as React from 'react'
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
import { IconGitHub } from '@/components/ui/icons'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group'

export const createGitProviderSchema = z.object({
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

export const updateGitProviderSchema = createGitProviderSchema.extend({
  provider: createGitProviderSchema.shape.provider.optional()
})

export type CreateGitProviderFormValues = z.infer<
  typeof createGitProviderSchema
>
export type UpdateGitProviderFormValues = z.infer<
  typeof updateGitProviderSchema
>

interface GitProviderFormProps {
  isNew?: boolean
  defaultValues?: Partial<z.infer<typeof createGitProviderSchema>>
  footer: React.ReactNode
  onSubmit: (values: any) => Promise<any>
}

export const GitProviderForm = React.forwardRef<
  {
    form: UseFormReturn<
      CreateGitProviderFormValues | UpdateGitProviderFormValues
    >
  },
  GitProviderFormProps
>(({ isNew, defaultValues, footer, onSubmit }, ref) => {
  const formSchema = isNew ? createGitProviderSchema : updateGitProviderSchema
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
          {isNew && (
            <FormField
              control={form.control}
              name="provider"
              disabled={!isNew}
              render={({ field: { onChange, ...rest } }) => (
                <FormItem>
                  <FormLabel required>Choose Git provider</FormLabel>
                  <FormControl>
                    <RadioGroup
                      className="flex flex-wrap gap-6"
                      orientation="horizontal"
                      onValueChange={onChange}
                      {...rest}
                    >
                      <div className="flex items-center space-x-2">
                        <RadioGroupItem value="github" id="github" />
                        <Label
                          className="flex cursor-pointer items-center gap-1"
                          htmlFor="github"
                        >
                          <IconGitHub className="h-6 w-6" />
                          <span>GitHub.com</span>
                        </Label>
                      </div>
                    </RadioGroup>
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
          )}
          <FormField
            control={form.control}
            name="displayName"
            render={({ field }) => (
              <FormItem>
                <FormLabel required>Display name</FormLabel>
                <FormDescription>
                  A display name to help identifying among different configs
                  using the same Git provider.
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
                <FormLabel required={isNew}>Access Token</FormLabel>
                <FormDescription>
                  <div>
                    Create a dedicated service user and generate a fine-grained
                    personal access token with the member role for the
                    organization or all projects to be managed.
                  </div>
                  <div className="ml-2">• Contents (Read-only)</div>
                </FormDescription>
                <FormControl>
                  <Input
                    placeholder="e.g. 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0NB"
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
          {/* <div className="flex items-center justify-between">
            <div>
              <FormMessage />
            </div>
            <div>
              <Button
                type="submit"
                disabled={isSubmitting || (!isNew && !isDirty)}
              >
                {isSubmitting && <IconSpinner className="mr-2" />}
                {isNew ? 'Create' : 'Update'}
              </Button>
            </div>
          </div> */}
        </form>
      </div>
    </Form>
  )
})

GitProviderForm.displayName = 'GitProviderForm'
