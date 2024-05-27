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

const deleteGithubSelfHostedRepositoryProviderMutation = graphql(/* GraphQL */ `
  mutation DeleteGithubSelfHostedRepositoryProvider($id: ID!) {
    deleteGithubSelfHostedRepositoryProvider(id: $id)
  }
`)

const updateGithubSelfHostedRepositoryProviderMutation = graphql(/* GraphQL */ `
  mutation UpdateGithubSelfHostedRepositoryProvider(
    $input: UpdateSelfHostedRepositoryProviderInput!
  ) {
    updateGithubSelfHostedRepositoryProvider(input: $input)
  }
`)

const deleteGitlabSelfHostedRepositoryProviderMutation = graphql(/* GraphQL */ `
  mutation DeleteGitlabSelfHostedRepositoryProvider($id: ID!) {
    deleteGitlabSelfHostedRepositoryProvider(id: $id)
  }
`)

const updateGitlabSelfHostedRepositoryProviderMutation = graphql(/* GraphQL */ `
  mutation UpdateGitlabSelfHostedRepositoryProvider(
    $input: UpdateSelfHostedRepositoryProviderInput!
  ) {
    updateGitlabSelfHostedRepositoryProvider(input: $input)
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

  const deleteRepositoryProvider = useMutation(
    deleteGithubRepositoryProviderMutation
  )

  const updateRepositoryProvider = useMutation(
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
    const res = await updateRepositoryProvider({
      input: {
        id,
        ...values
      }
    })
    // todo update resolver
    if (res?.data?.updateGithubRepositoryProvider) {
      onUpdate?.()
    }
  }

  const handleDeleteRepositoryProvider = async () => {
    const res = await deleteRepositoryProvider({ id })
    // todo delete resovler
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
