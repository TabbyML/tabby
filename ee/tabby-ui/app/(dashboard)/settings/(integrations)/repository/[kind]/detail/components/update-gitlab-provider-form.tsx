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

  const onSubmit = async (values: CreateRepositoryProviderFormValues) => {
    const res = await updateGitlabRepositoryProvider({
      input: {
        id,
        ...values
      }
    })
    if (res?.data?.updateGitlabRepositoryProvider) {
      onUpdate?.()
    }
  }

  const handleDeleteRepositoryProvider = async () => {
    const res = await deleteGitlabRepositoryProvider({ id })
    if (res?.data?.deleteGitlabRepositoryProvider) {
      onDelete?.()
    } else {
      toast.error(
        res?.error?.message || 'Failed to delete GitLab repository provider'
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
