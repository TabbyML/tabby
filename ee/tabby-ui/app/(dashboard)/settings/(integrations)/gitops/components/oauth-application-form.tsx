'use client'

import React from 'react'
import type { UseFormReturn } from 'react-hook-form'
import * as z from 'zod'

import { useExternalURL } from '@/lib/hooks/use-network-setting'
import { cn } from '@/lib/utils'
import {
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage
} from '@/components/ui/form'
import { Input } from '@/components/ui/input'
import { CopyButton } from '@/components/copy-button'

export const oauthInfoFormSchema = z.object({
  applicationId: z.string(),
  secret: z.string()
})

export type OAuthApplicationFormValues = z.infer<typeof oauthInfoFormSchema>

interface OAuthApplicationFormProps
  extends React.HTMLAttributes<HTMLDivElement> {
  form: UseFormReturn
}

export const OAuthApplicationForm = React.forwardRef<
  HTMLDivElement,
  OAuthApplicationFormProps
>(({ className, form }, ref) => {
  const externalURL = useExternalURL()
  const integrationsCallbackURL = externalURL
    ? `${externalURL}/integrations/github/callback`
    : ''

  return (
    <div className={cn('grid gap-6', className)} ref={ref}>
      {!!integrationsCallbackURL && (
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
                {integrationsCallbackURL}
                <CopyButton type="button" value={integrationsCallbackURL} />
              </span>
            </div>
          </div>
        </FormItem>
      )}
      <FormField
        control={form.control}
        name="applicationId"
        render={({ field }) => (
          <FormItem>
            <FormLabel required>Application ID</FormLabel>
            <FormControl>
              <Input
                placeholder="e.g. ae1542c44b154c10c859"
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
        name="secret"
        render={({ field }) => (
          <FormItem>
            <FormLabel required>Application secret</FormLabel>
            <FormControl>
              <Input
                placeholder="e.g. e363c08d7e9ca4e66e723a53f38a21f6a54c1b83"
                autoCapitalize="none"
                autoCorrect="off"
                {...field}
              />
            </FormControl>
            <FormMessage />
          </FormItem>
        )}
      />
    </div>
  )
})

OAuthApplicationForm.displayName = 'OAuthApplicationForm'
