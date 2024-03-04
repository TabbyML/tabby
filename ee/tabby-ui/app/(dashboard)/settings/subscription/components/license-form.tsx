'use client'

import * as React from 'react'
import { zodResolver } from '@hookform/resolvers/zod'
import { useForm } from 'react-hook-form'
import { toast } from 'sonner'
import * as z from 'zod'

import { graphql } from '@/lib/gql/generates'
import { useDebounceCallback } from '@/lib/hooks/use-debounce'
import { useMutation } from '@/lib/tabby/gql'
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
  canReset?: boolean
}

const uploadLicenseMutation = graphql(/* GraphQL */ `
  mutation UploadLicense($license: String!) {
    uploadLicense(license: $license)
  }
`)

const resetLicenseMutation = graphql(/* GraphQL */ `
  mutation ResetLicense {
    resetLicense
  }
`)

export function LicenseForm({
  className,
  onSuccess,
  canReset,
  ...props
}: LicenseFormProps) {
  const form = useForm<FormValues>({
    resolver: zodResolver(formSchema)
  })
  const license = form.watch('license')
  const [isSubmitting, setIsSubmitting] = React.useState(false)
  const [resetDialogOpen, setResetDialogOpen] = React.useState(false)
  const [isResetting, setIsResetting] = React.useState(false)

  const toggleSubmitting = useDebounceCallback(
    (value: boolean, success?: boolean) => {
      setIsSubmitting(value)
      if (success) {
        form.reset({ license: '' })
        toast.success('License is uploaded')
        onSuccess?.()
      }
    },
    500,
    { leading: true }
  )

  const toggleResetting = useDebounceCallback(
    (value: boolean, success?: boolean) => {
      setIsResetting(value)
      if (success) {
        setResetDialogOpen(false)
        onSuccess?.()
      }
    },
    500,
    { leading: true }
  )

  const uploadLicense = useMutation(uploadLicenseMutation, {
    form
  })

  const resetLicense = useMutation(resetLicenseMutation)

  const onSubmit = (values: FormValues) => {
    toggleSubmitting.run(true)
    return uploadLicense(values).then(res => {
      toggleSubmitting.run(false, res?.data?.uploadLicense)
    })
  }

  const onReset: React.MouseEventHandler<HTMLButtonElement> = e => {
    e.preventDefault()
    toggleResetting.run(true)
    resetLicense().then(res => {
      const isSuccess = res?.data?.resetLicense
      toggleResetting.run(false, isSuccess)

      if (res?.error) {
        toast.error(res.error.message ?? 'reset failed')
      }
    })
  }

  const onResetDialogOpenChange = (v: boolean) => {
    if (isResetting) return
    setResetDialogOpen(v)
  }

  return (
    <div className={cn(className)} {...props}>
      <Form {...form}>
        <form className="grid gap-6" onSubmit={form.handleSubmit(onSubmit)}>
          <FormField
            control={form.control}
            name="license"
            render={({ field }) => (
              <FormItem>
                <FormControl>
                  <Textarea
                    className="min-h-[200px]"
                    placeholder="Paste your license here - write only"
                    {...field}
                  />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
          <div className="flex items-start justify-between gap-4">
            <div>
              <FormMessage />
            </div>
            <div className="flex shrink-0 items-center gap-4">
              <AlertDialog
                open={resetDialogOpen}
                onOpenChange={onResetDialogOpenChange}
              >
                {canReset && (
                  <AlertDialogTrigger asChild>
                    <Button type="button" variant="hover-destructive">
                      Reset
                    </Button>
                  </AlertDialogTrigger>
                )}
                <AlertDialogContent>
                  <AlertDialogHeader>
                    <AlertDialogTitle>
                      Are you absolutely sure?
                    </AlertDialogTitle>
                    <AlertDialogDescription>
                      This action cannot be undone. It will reset the current
                      license.
                    </AlertDialogDescription>
                  </AlertDialogHeader>
                  <AlertDialogFooter>
                    <AlertDialogCancel>Cancel</AlertDialogCancel>
                    <AlertDialogAction
                      className={buttonVariants({ variant: 'destructive' })}
                      onClick={onReset}
                      disabled={isResetting}
                    >
                      {isResetting && (
                        <IconSpinner className="mr-2 h-4 w-4 animate-spin" />
                      )}
                      Yes, reset it
                    </AlertDialogAction>
                  </AlertDialogFooter>
                </AlertDialogContent>
              </AlertDialog>
              <Button type="submit" disabled={isSubmitting || !license}>
                {isSubmitting && (
                  <IconSpinner className="mr-2 h-4 w-4 animate-spin" />
                )}
                Upload License
              </Button>
            </div>
          </div>
        </form>
      </Form>
    </div>
  )
}
