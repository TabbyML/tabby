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

const deleteGithubRepositoryProviderMutation = graphql(/* GraphQL */ `
  mutation DeleteGithubRepositoryProvider($id: ID!) {
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

  const onSubmit = async (values: RepositoryProviderFormValues) => {
    await updateGithubRepositoryProvider({
      input: {
        id,
        ...values
      }
    })
  }

  const handleDeleteRepositoryProvider = async () => {
    const res = await deleteGithubRepositoryProvider({ id })
    if (res?.data?.deleteGithubRepositoryProvider) {
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
