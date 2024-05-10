'use client'

import React from 'react'
import { toast } from 'sonner'

import { graphql } from '@/lib/gql/generates'
import { useMutation } from '@/lib/tabby/gql'

import {
  CommonProviderForm,
  CreateRepositoryProviderFormValues,
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
  defaultValues?: Partial<CreateRepositoryProviderFormValues>
  onSuccess?: () => void
  onDelete: () => void
  onUpdate: () => void
}

export const UpdateProviderForm: React.FC<UpdateProviderFormProps> = ({
  defaultValues,
  onSuccess,
  onDelete,
  onUpdate,
  id
}) => {
  const form = useRepositoryProviderForm(false, defaultValues)

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

  const onSubmit = async (values: CreateRepositoryProviderFormValues) => {
    const res = await updateGithubRepositoryProvider({
      input: {
        id,
        ...values
      }
    })
    if (res?.data?.updateGithubRepositoryProvider) {
      onUpdate?.()
    } else {
      toast.error(
        res?.error?.message || 'Failed to update GitHub repository provider'
      )
    }
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
    <CommonProviderForm
      onSubmit={onSubmit}
      onDelete={handleDeleteRepositoryProvider}
      deletable
      cancleable={false}
      form={form}
      isNew={false}
    />
  )
}
