'use client'

import React from 'react'
import { zodResolver } from '@hookform/resolvers/zod'
import { isEmpty, trim } from 'lodash-es'
import { useForm } from 'react-hook-form'
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
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage
} from '@/components/ui/form'
import { IconSpinner } from '@/components/ui/icons'
import { Input } from '@/components/ui/input'

const deleteGithubRepositoryProviderMutation = graphql(/* GraphQL */ `
  mutation DeleteGithubRepositoryProvider($id: ID!) {
    deleteGithubRepositoryProvider(id: $id)
  }
`)

const updateGithubRepositoryProviderMutation = graphql(/* GraphQL */ `
  mutation UpdateGithubRepositoryProvider(
    $id: ID!
    $applicationId: String!
    $secret: String
  ) {
    updateGithubRepositoryProvider(
      id: $id
      applicationId: $applicationId
      secret: $secret
    )
  }
`)

export const formSchema = z.object({
  applicationId: z.string(),
  secret: z.string().optional()
})

type FormValues = z.infer<typeof formSchema>

interface UpdateProviderFormProps {
  id: string
  defaultValues?: Partial<FormValues>
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
  const [deleteAlertVisible, setDeleteAlertVisible] = React.useState(false)
  const [isDeleting, setIsDeleting] = React.useState(false)
  const form = useForm<FormValues>({
    resolver: zodResolver(formSchema),
    defaultValues
  })

  const deleteGithubRepositoryProvider = useMutation(
    deleteGithubRepositoryProviderMutation
  )

  const isDirty = !isEmpty(form.formState.dirtyFields)

  const updateGithubRepositoryProvider = useMutation(
    updateGithubRepositoryProviderMutation,
    {
      form,
      onCompleted(values) {
        if (values?.updateGithubRepositoryProvider) {
          toast.success('Updated repository provider successfully')
          onSuccess?.()
        }
      }
    }
  )

  const onSubmit = async (values: FormValues) => {
    await updateGithubRepositoryProvider({
      id,
      applicationId: values.applicationId,
      secret: trim(values.secret) || undefined
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
                <FormLabel>Application secret</FormLabel>
                <FormControl>
                  <Input
                    placeholder="*****"
                    autoCapitalize="none"
                    autoCorrect="off"
                    {...field}
                  />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
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
