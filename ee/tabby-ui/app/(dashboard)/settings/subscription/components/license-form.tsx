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


const passwordResetMutation = graphql(/* GraphQL */ `
  mutation passwordReset($input: PasswordResetInput!) {
    passwordReset(input: $input)
  }
`)

const formSchema = z.object({
  license: z.string()
})

type FormValues = z.infer<typeof formSchema>

interface LicenseFormProps extends React.HTMLAttributes<HTMLDivElement> {
  code?: string
  onSuccess?: () => void
}

export function LicenseForm({
  className,
  code,
  onSuccess,
  ...props
}: LicenseFormProps) {
  const form = useForm<FormValues>({
    resolver: zodResolver(formSchema)
  })
  const license = form.watch('license')
  const { isSubmitting } = form.formState

  // todo api
  const passwordReset = useMutation(passwordResetMutation, {
    form
  })

  const onSubmit = (values: FormValues) => {
    // todo
    // return passwordReset({
    //   input: {
    //     ...values,
    //     code: values.code ?? ''
    //   }
    // }).then(res => {
    //   if (res?.data?.passwordReset) {
    //     onSuccess?.()
    //   }
    // })
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
                  <Textarea placeholder='Paste your license here' {...field} />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
          <div className='text-right'>
            <Button type="submit" className="mt-2" disabled={isSubmitting || !license}>
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
