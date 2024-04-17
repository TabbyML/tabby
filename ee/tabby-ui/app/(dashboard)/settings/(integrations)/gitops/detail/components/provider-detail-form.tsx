'use client'

import React from 'react'
import { useRouter } from 'next/navigation'
import { zodResolver } from '@hookform/resolvers/zod'
import { isEmpty } from 'lodash-es'
import { useForm, UseFormReturn } from 'react-hook-form'
import { toast } from 'sonner'
import * as z from 'zod'

import { graphql } from '@/lib/gql/generates'
import { useMutation } from '@/lib/tabby/gql'
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
  basicInfoFormSchema,
  BasicInfoFormValues
} from '../../components/basic-info-form'
import {
  OAuthApplicationForm,
  OAuthApplicationFormValues,
  oauthInfoFormSchema
} from '../../components/oauth-application-form'

const deleteGithubRepositoryProviderMutation = graphql(/* GraphQL */ `
  mutation DeleteGithubRepositoryProvider($id: ID!) {
    deleteGithubRepositoryProvider(id: $id)
  }
`)

const updateProviderFormSchema = z.union([
  basicInfoFormSchema,
  oauthInfoFormSchema
])

type UpdateProviderFormValues = z.infer<typeof updateProviderFormSchema>

interface UpdateProviderFormProps {
  id: string
  defaultValues?: Partial<UpdateProviderFormValues>
  onSuccess?: () => void
  onDelete: () => void
  onBack: () => void
}

export const UpdateProviderForm: React.FC<UpdateProviderFormProps> = ({
  defaultValues,
  onSuccess,
  onDelete,
  onBack,
  id
}) => {
  const ENCODE_PASSWORD = '********************************'

  const router = useRouter()
  const [deleteAlertVisible, setDeleteAlertVisible] = React.useState(false)
  const [isDeleting, setIsDeleting] = React.useState(false)
  const form = useForm<UpdateProviderFormValues>({
    resolver: zodResolver(updateProviderFormSchema),
    defaultValues: { ...defaultValues, applicationSecret: ENCODE_PASSWORD }
  })

  const deleteGithubRepositoryProvider = useMutation(
    deleteGithubRepositoryProviderMutation
  )

  const isDirty = !isEmpty(form.formState.dirtyFields)

  const onSubmit = async () => {
    // todo update
    onSuccess?.()
  }

  const handleDeleteRepositoryProvider: React.MouseEventHandler<
    HTMLButtonElement
  > = async e => {
    e.preventDefault()
    setIsDeleting(true)
    try {
      const res = await deleteGithubRepositoryProvider({ id })
      if (res?.data?.deleteGithubRepositoryProvider) {
        toast.success('Deleted repository provider successfully')
        onDelete?.()
      } else {
        toast.error(
          res?.error?.message || 'Failed to delete repository provider'
        )
      }
    } catch (error) {
      toast.error('Failed to delete repository provider')
    } finally {
      setIsDeleting(false)
    }
  }

  return (
    <Form {...form}>
      <div className="grid gap-2">
        <form className="grid gap-6" onSubmit={form.handleSubmit(onSubmit)}>
          <BasicInfoForm
            form={form as UseFormReturn<BasicInfoFormValues>}
            isUpdate
          />
          <OAuthApplicationForm
            form={form as UseFormReturn<OAuthApplicationFormValues>}
          />
          <div className="flex justify-between">
            <Button variant="ghost" onClick={() => onBack()}>
              Back to providers
            </Button>
            <div className="items-cetner flex gap-4">
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
                      onClick={handleDeleteRepositoryProvider}
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
