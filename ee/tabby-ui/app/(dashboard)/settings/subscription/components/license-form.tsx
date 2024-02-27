'use client'

import * as React from 'react'
import { zodResolver } from '@hookform/resolvers/zod'
import { useForm } from 'react-hook-form'
import { toast } from 'sonner'
import * as z from 'zod'

import { graphql } from '@/lib/gql/generates'
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
  canDelete?: boolean
}

const uploadLicenseMutation = graphql(/* GraphQL */ `
  mutation UploadLicense($license: String!) {
    uploadLicense(license: $license)
  }
`)

const deleteLicenseMutation = graphql(/* GraphQL */ `
  mutation DeleteLicense {
    deleteLicense
  }
`)

export function LicenseForm({
  className,
  onSuccess,
  canDelete,
  ...props
}: LicenseFormProps) {
  const form = useForm<FormValues>({
    resolver: zodResolver(formSchema)
  })
  const license = form.watch('license')
  const { isSubmitting } = form.formState
  const [isDeleting, setIsDeleting] = React.useState(false)
  const [deleteDialogOpen, setDeleteDialogOpen] = React.useState(false)

  const uploadLicense = useMutation(uploadLicenseMutation, {
    form
  })

  const deleteLicense = useMutation(deleteLicenseMutation)

  const onSubmit = (values: FormValues) => {
    return uploadLicense(values).then(res => {
      if (res?.data?.uploadLicense) {
        form.reset({ license: '' })
        toast.success('License is uploaded')
        onSuccess?.()
      }
    })
  }

  const onDelete: React.MouseEventHandler<HTMLButtonElement> = e => {
    e.preventDefault()
    setIsDeleting(true)
    deleteLicense()
      .then(res => {
        if (res?.data?.deleteLicense) {
          setDeleteDialogOpen(false)
          onSuccess?.()
        } else if (res?.error) {
          toast.error(res.error.message ?? 'delete failed')
        }
      })
      .finally(() => {
        setIsDeleting(false)
      })
  }

  const onDeleteDialogOpenChange = (v: boolean) => {
    if (isDeleting) return
    setDeleteDialogOpen(v)
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
          <div className="mt-2 flex items-center justify-end gap-4">
            <AlertDialog
              open={deleteDialogOpen}
              onOpenChange={onDeleteDialogOpenChange}
            >
              {canDelete && (
                <AlertDialogTrigger asChild>
                  <Button type="button" variant="hover-destructive">
                    Delete
                  </Button>
                </AlertDialogTrigger>
              )}
              <AlertDialogContent>
                <AlertDialogHeader>
                  <AlertDialogTitle>Are you absolutely sure?</AlertDialogTitle>
                  <AlertDialogDescription>
                    This action cannot be undone. It will delete the current
                    license.
                  </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                  <AlertDialogCancel>Cancel</AlertDialogCancel>
                  <AlertDialogAction
                    className={buttonVariants({ variant: 'destructive' })}
                    onClick={onDelete}
                    disabled={isDeleting}
                  >
                    {isDeleting && (
                      <IconSpinner className="mr-2 h-4 w-4 animate-spin" />
                    )}
                    Yes, delete it
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
        </form>
        <FormMessage className="text-center" />
      </Form>
    </div>
  )
}
