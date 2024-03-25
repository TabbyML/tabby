'use client'

import React from 'react'
import type { UseFormReturn } from 'react-hook-form'
import * as z from 'zod'

import { cn } from '@/lib/utils'
import {
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage
} from '@/components/ui/form'
import { Input } from '@/components/ui/input'

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
  return (
    <div className={cn('grid gap-6', className)} ref={ref}>
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
            <FormLabel required>Secret</FormLabel>
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
