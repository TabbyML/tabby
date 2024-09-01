'use client'

import React from 'react'
import { toast } from 'sonner'

import { graphql } from '@/lib/gql/generates'
import { IntegrationKind } from '@/lib/gql/generates/graphql'
import { useMutation } from '@/lib/tabby/gql'

import {
  CommonProviderForm,
  CreateRepositoryProviderFormValues,
  useRepositoryProviderForm
} from '../../components/common-provider-form'

const updateIntegrationMutation = graphql(/* GraphQL */ `
  mutation UpdateIntegration($input: UpdateIntegrationInput!) {
    updateIntegration(input: $input)
  }
`)

const deleteIntegrationMutation = graphql(/* GraphQL */ `
  mutation DeleteIntegration($id: ID!, $kind: IntegrationKind!) {
    deleteIntegration(id: $id, kind: $kind)
  }
`)

interface UpdateProviderFormProps {
  id: string
  kind: IntegrationKind
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
  id,
  kind
}) => {
  const form = useRepositoryProviderForm(false, kind, defaultValues)

  const deleteRepositoryProvider = useMutation(deleteIntegrationMutation)

  const updateRepositoryProvider = useMutation(updateIntegrationMutation, {
    form
  })

  const onSubmit = async (values: CreateRepositoryProviderFormValues) => {
    const res = await updateRepositoryProvider({
      input: {
        id,
        ...values,
        kind
      }
    })
    if (res?.data?.updateIntegration) {
      toast.success('Updated provider successfully')
      form?.reset(form?.getValues())
      onSuccess?.()
      onUpdate?.()
    }
  }

  const handleDeleteRepositoryProvider = async () => {
    const res = await deleteRepositoryProvider({ id, kind })
    if (res?.data?.deleteIntegration) {
      onDelete?.()
    } else {
      toast.error(res?.error?.message || 'Failed to delete provider')
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
