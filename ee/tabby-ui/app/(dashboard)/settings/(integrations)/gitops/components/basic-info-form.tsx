'use client'

import * as React from 'react'
import { UseFormReturn } from 'react-hook-form'
import * as z from 'zod'

import { cn } from '@/lib/utils'
import {
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

export const basicInfoFormSchema = z.object({
  displayName: z
    .string()
    .trim()
    .regex(
      /^[\w-]+$/,
      'Display name must contain only alphanumeric characters, underscores, and hyphens'
    ),
  provider: z.string()
})

export type BasicInfoFormValues = z.infer<typeof basicInfoFormSchema>

interface BasicInfoFormProps extends React.HTMLAttributes<HTMLDivElement> {
  form: UseFormReturn<BasicInfoFormValues>
  isUpdate?: boolean
}

export const BasicInfoForm = React.forwardRef<
  HTMLDivElement,
  BasicInfoFormProps
>(({ className, form, isUpdate, ...rest }, ref) => {
  return (
    <div className={cn('grid gap-6', className)} ref={ref} {...rest}>
      <FormField
        control={form.control}
        name="provider"
        disabled={isUpdate}
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
      <FormField
        control={form.control}
        name="displayName"
        render={({ field }) => (
          <FormItem>
            <FormLabel required>Display name</FormLabel>
            <FormDescription>
              A display name to help identifying among different configs using
              the same Git provider.
            </FormDescription>
            <FormControl>
              <Input
                placeholder="e.g. GitHub"
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

BasicInfoForm.displayName = 'BasicInfoForm'
