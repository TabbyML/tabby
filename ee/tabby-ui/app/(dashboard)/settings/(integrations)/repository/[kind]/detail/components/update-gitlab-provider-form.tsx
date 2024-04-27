'use client'

import React from 'react'
import { toast } from 'sonner'

import { graphql } from '@/lib/gql/generates'
import { useMutation } from '@/lib/tabby/gql'

import {
  ProviderForm,
  RepositoryProviderFormValues,
  useRepositoryProviderForm
} from '../../components/common-provider-form'

const deleteGitlabRepositoryProviderMutation = graphql(/* GraphQL */ `
  mutation DeleteGitlabRepositoryProvider($id: ID!) {
    deleteGitlabRepositoryProvider(id: $id)
  }
`)

const updateGitlabRepositoryProviderMutation = graphql(/* GraphQL */ `
  mutation UpdateGitlabRepositoryProvider(
    $input: UpdateRepositoryProviderInput!
  ) {
    updateGitlabRepositoryProvider(input: $input)
  }
`)

interface UpdateProviderFormProps {
  id: string
  defaultValues?: Partial<RepositoryProviderFormValues>
  onSuccess?: () => void
  onDelete: () => void
}

export const UpdateProviderForm: React.FC<UpdateProviderFormProps> = ({
  defaultValues,
  onSuccess,
  onDelete,
  id
}) => {
  const form = useRepositoryProviderForm(defaultValues)

  const deleteGitlabRepositoryProvider = useMutation(
    deleteGitlabRepositoryProviderMutation
  )

  const updateGitlabRepositoryProvider = useMutation(
    updateGitlabRepositoryProviderMutation,
    {
      form,
      onCompleted(values) {
        if (values?.updateGitlabRepositoryProvider) {
          toast.success('Updated GitLab repository provider successfully')
          form?.reset(form?.getValues())
          onSuccess?.()
        }
      }
    }
  )

  const onSubmit = async (values: RepositoryProviderFormValues) => {
    await updateGitlabRepositoryProvider({
      input: {
        id,
        ...values
      }
    })
  }

  const handleDeleteRepositoryProvider = async () => {
    const res = await deleteGitlabRepositoryProvider({ id })
    if (res?.data?.deleteGitlabRepositoryProvider) {
      onDelete?.()
    } else {
      toast.error(
        res?.error?.message || 'Failed to delete GitHub repository provider'
      )
    }
  }

  return (
    <ProviderForm
      onSubmit={onSubmit}
      onDelete={handleDeleteRepositoryProvider}
      deletable
      cancleable={false}
      form={form}
    />
  )
}
