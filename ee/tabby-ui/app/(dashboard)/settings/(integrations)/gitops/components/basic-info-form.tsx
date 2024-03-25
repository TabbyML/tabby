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
import { IconGitHub, IconGitLab } from '@/components/ui/icons'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group'

export const basicInfoFormSchema = z.object({
  instanceUrl: z.string(),
  displayName: z.string(),
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
  const provider = form.watch('provider')
  let instanceUrlLabelPrefix = 'GitHub'
  if (provider?.startsWith('gitlab')) {
    instanceUrlLabelPrefix = 'GitLab'
  }

  React.useEffect(() => {
    if (provider === 'github') {
      form.setValue('instanceUrl', 'https://github.com')
    } else if (provider === 'gitlab') {
      form.setValue('instanceUrl', 'https://gitlab.com')
    } else {
      form.setValue('instanceUrl', '')
    }
  }, [provider])

  return (
    <div className={cn('grid gap-6', className)} ref={ref} {...rest}>
      {!isUpdate && (
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
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem
                      value="github_self_hosted"
                      id="github_self_hosted"
                    />
                    <Label
                      className="flex cursor-pointer items-center gap-1"
                      htmlFor="github_self_hosted"
                    >
                      <IconGitHub className="h-6 w-6" />
                      <span>GitHub Self-Hosted Enterprise Edition (EE)</span>
                    </Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem
                      value="gitlab_self_hosted"
                      id="gitlab_self_hosted"
                    />
                    <Label
                      className="flex cursor-pointer items-center gap-1"
                      htmlFor="gitlab_self_hosted"
                    >
                      <IconGitLab className="h-6 w-6" />
                      <span>GitLab Self-Hosted</span>
                    </Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="gitlab" id="gitlab" />
                    <Label
                      className="flex cursor-pointer items-center gap-1"
                      htmlFor="gitlab"
                    >
                      <IconGitLab className="h-6 w-6" />
                      <span>GitLab.com</span>
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
        name="instanceUrl"
        disabled={['github', 'gitlab'].includes(provider)}
        render={({ field }) => (
          <FormItem>
            <FormLabel required>
              {instanceUrlLabelPrefix} instance URL
            </FormLabel>
            <FormDescription>
              The VCS instance URL. Make sure this instance and Tabby are
              network reachable from each other.
            </FormDescription>
            <FormControl>
              <Input
                placeholder="e.g. https://github.yourcompany.com"
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
        name="displayName"
        render={({ field }) => (
          <FormItem>
            <FormLabel>Display name</FormLabel>
            <FormDescription>
              An optional display name to help identifying among different
              configs using the same Git provider.
            </FormDescription>
            <FormControl>
              <Input
                placeholder="e.g. GitHub.com"
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
