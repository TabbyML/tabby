'use client'

import React from 'react'
import { useRouter } from 'next/navigation'
import { zodResolver } from '@hookform/resolvers/zod'
import { isEmpty } from 'lodash-es'
import { useForm } from 'react-hook-form'
import * as z from 'zod'

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
import { Form, FormMessage } from '@/components/ui/form'
import { IconSpinner } from '@/components/ui/icons'

import {
  BasicInfoForm,
  basicInfoFormSchema
} from '../../components/basic-info-form'
import {
  OAuthApplicationForm,
  oauthInfoFormSchema
} from '../../components/oauth-application-form'

const updateProviderFormSchema = z.union([
  basicInfoFormSchema,
  oauthInfoFormSchema
])

type UpdateProviderFormValues = z.infer<typeof updateProviderFormSchema>

interface UpdateProviderFormProps {
  defaultValues?: Partial<UpdateProviderFormValues>
  onSuccess?: () => void
  onDelete: () => void
  onBack: () => void
}

export const UpdateProviderForm: React.FC<UpdateProviderFormProps> = ({
  defaultValues,
  onSuccess,
  onDelete,
  onBack
}) => {
  const ENCODE_PASSWORD = '********************************'

  const router = useRouter()
  const [deleteAlertVisible, setDeleteAlertVisible] = React.useState(false)
  const [isDeleting, setIsDeleting] = React.useState(false)
  const form = useForm<UpdateProviderFormValues>({
    resolver: zodResolver(updateProviderFormSchema),
    defaultValues: { ...defaultValues, secret: ENCODE_PASSWORD }
  })

  const isDirty = !isEmpty(form.formState.dirtyFields)

  const onSubmit = async () => {
    // todo update
    onSuccess?.()
  }

  return (
    <Form {...form}>
      <div className="grid gap-2">
        <form className="grid gap-6" onSubmit={form.handleSubmit(onSubmit)}>
          <BasicInfoForm form={form} isUpdate />
          <OAuthApplicationForm form={form} />
          <div className="flex justify-between">
            <Button variant="ghost" onClick={() => onBack()}>
              Back to providers
            </Button>
            <div className="flex items-cetner gap-4">
              <AlertDialog
                open={deleteAlertVisible}
                onOpenChange={setDeleteAlertVisible}
              >
                <AlertDialogTrigger asChild>
                  <Button type="button" variant="hover-destructive">
                    Delete
                  </Button>
                </AlertDialogTrigger>
                <AlertDialogContent>
                  <AlertDialogHeader>
                    <AlertDialogTitle>
                      Are you absolutely sure?
                    </AlertDialogTitle>
                    <AlertDialogDescription>
                      This will delete the provider and unlink any repositories
                      that have already been linked to the provider.
                    </AlertDialogDescription>
                  </AlertDialogHeader>
                  <AlertDialogFooter>
                    <AlertDialogCancel>Cancel</AlertDialogCancel>
                    <AlertDialogAction
                      className={buttonVariants({ variant: 'destructive' })}
                      onClick={() => onDelete()}
                    >
                      {isDeleting && (
                        <IconSpinner className="mr-2 h-4 w-4 animate-spin" />
                      )}
                      Yes, delete it
                    </AlertDialogAction>
                  </AlertDialogFooter>
                </AlertDialogContent>
              </AlertDialog>
              <Button type="submit" disabled={!isDirty}>
                Update
              </Button>
            </div>
          </div>
        </form>
        <FormMessage className="text-center" />
      </div>
    </Form>
  )
}
