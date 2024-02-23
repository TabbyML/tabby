'use client'

import * as React from 'react'
import { zodResolver } from '@hookform/resolvers/zod'
import { useForm } from 'react-hook-form'
import * as z from 'zod'

import { graphql } from '@/lib/gql/generates'
import { useMutation } from '@/lib/tabby/gql'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormMessage
} from '@/components/ui/form'
import { IconSpinner } from '@/components/ui/icons'
import { Textarea } from '@/components/ui/textarea'

const formSchema = z.object({
  license: z.string()
})

type FormValues = z.infer<typeof formSchema>

interface LicenseFormProps extends React.HTMLAttributes<HTMLDivElement> {
  onSuccess?: () => void
}

const uploadLicenseMutation = graphql(/* GraphQL */ `
  mutation UploadLicense($license: String!) {
    uploadLicense(license: $license)
  }
`)

export function LicenseForm({
  className,
  onSuccess,
  ...props
}: LicenseFormProps) {
  const form = useForm<FormValues>({
    resolver: zodResolver(formSchema)
  })
  const license = form.watch('license')
  const { isSubmitting } = form.formState

  const uploadLicense = useMutation(uploadLicenseMutation, {
    form
  })

  const onSubmit = (values: FormValues) => {
    return uploadLicense(values).then(res => {
      if (res?.data?.uploadLicense) {
        form.reset({ license: "" });
        onSuccess?.()
      }
    })
  }

  return (
    <div className={cn('grid gap-6', className)} {...props}>
      <Form {...form}>
        <form className="grid gap-2" onSubmit={form.handleSubmit(onSubmit)}>
          <FormField
            control={form.control}
            name="license"
            render={({ field }) => (
              <FormItem>
                <FormControl>
                  <Textarea className="min-h-[200px]" placeholder="Paste your license here - write only" {...field} />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
          <div className="text-right">
            <Button
              type="submit"
              className="mt-2"
              disabled={isSubmitting || !license}
            >
              {isSubmitting && (
                <IconSpinner className="mr-2 h-4 w-4 animate-spin" />
              )}
              Upload License
            </Button>
          </div>
        </form>
        <FormMessage className="text-center" />
      </Form>
    </div>
  )
}
