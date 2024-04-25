'use client'

import React from 'react'
import { isEmpty } from 'lodash-es'
import { UseFormReturn } from 'react-hook-form'
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
import { FormMessage } from '@/components/ui/form'
import { IconSpinner } from '@/components/ui/icons'

import {
  GithubProviderForm,
  UpdateGithubProviderFormValues,
  updateGithubProviderSchema
} from '../../components/github-form'

const deleteGithubRepositoryProviderMutation = graphql(/* GraphQL */ `
  mutation DeleteRepositoryProvider($id: ID!) {
    deleteGithubRepositoryProvider(id: $id)
  }
`)

const updateGithubRepositoryProviderMutation = graphql(/* GraphQL */ `
  mutation UpdateGithubRepositoryProvider(
    $input: UpdateRepositoryProviderInput!
  ) {
    updateGithubRepositoryProvider(input: $input)
  }
`)

type FormValues = z.infer<typeof updateGithubProviderSchema>

interface UpdateProviderFormProps {
  id: string
  defaultValues?: Partial<FormValues>
  onSuccess?: () => void
  onDelete: () => void
}

export const UpdateProviderForm: React.FC<UpdateProviderFormProps> = ({
  defaultValues,
  onSuccess,
  onDelete,
  id
}) => {
  const formRef = React.useRef<{
    form: UseFormReturn<UpdateGithubProviderFormValues>
  }>(null)
  const [deleteAlertVisible, setDeleteAlertVisible] = React.useState(false)
  const [isDeleting, setIsDeleting] = React.useState(false)
  const form = formRef.current?.form
  const isSubmitting = form?.formState?.isSubmitting
  const isDirty = !isEmpty(form?.formState?.dirtyFields)

  const deleteGithubRepositoryProvider = useMutation(
    deleteGithubRepositoryProviderMutation
  )

  const updateGithubRepositoryProvider = useMutation(
    updateGithubRepositoryProviderMutation,
    {
      form,
      onCompleted(values) {
        if (values?.updateGithubRepositoryProvider) {
          toast.success('Updated GitHub repository provider successfully')
          form?.reset(form?.getValues())
          onSuccess?.()
        }
      }
    }
  )

  const onSubmit = async (values: FormValues) => {
    await updateGithubRepositoryProvider({
      input: {
        id,
        ...values
      }
    })
  }

  const handleDeleteRepositoryProvider: React.MouseEventHandler<
    HTMLButtonElement
  > = async e => {
    e.preventDefault()
    setIsDeleting(true)
    try {
      const res = await deleteGithubRepositoryProvider({ id })
      if (res?.data?.deleteGithubRepositoryProvider) {
        onDelete?.()
      } else {
        toast.error(
          res?.error?.message || 'Failed to delete GitHub repository provider'
        )
      }
    } catch (error) {
      toast.error('Failed to delete GitHub repository provider')
    } finally {
      setIsDeleting(false)
    }
  }

  return (
    <GithubProviderForm
      ref={formRef}
      defaultValues={defaultValues}
      footer={
        <div className="flex justify-between">
          <div>
            <FormMessage />
          </div>
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
                  <AlertDialogTitle>Are you absolutely sure?</AlertDialogTitle>
                  <AlertDialogDescription>
                    This will delete the provider and remove any repositories
                    that have already been added to the provider.
                  </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                  <AlertDialogCancel>Cancel</AlertDialogCancel>
                  <AlertDialogAction
                    className={buttonVariants({ variant: 'destructive' })}
                    onClick={handleDeleteRepositoryProvider}
                    disabled={isDeleting}
                  >
                    {isDeleting && <IconSpinner className="mr-2" />}
                    Yes, delete it
                  </AlertDialogAction>
                </AlertDialogFooter>
              </AlertDialogContent>
            </AlertDialog>
            <Button type="submit" disabled={!isDirty}>
              {isSubmitting && <IconSpinner className="mr-2 " />}
              Update
            </Button>
          </div>
        </div>
      }
      onSubmit={onSubmit}
    />
  )
}
