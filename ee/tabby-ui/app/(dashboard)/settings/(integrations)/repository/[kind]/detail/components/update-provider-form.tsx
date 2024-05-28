'use client'

import React from 'react'
import { toast } from 'sonner'

import { graphql } from '@/lib/gql/generates'
import { QueryResponseData, useMutation } from '@/lib/tabby/gql'

import {
  CommonProviderForm,
  CreateRepositoryProviderFormValues,
  useRepositoryProviderForm
} from '../../components/common-provider-form'
import { RepositoryKind } from '@/lib/gql/generates/graphql'
import { TypedDocumentNode } from 'urql'

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
  kind: RepositoryKind
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
  const form = useRepositoryProviderForm(false, defaultValues)

  const { updateMutation, deleteMutation, updateResolver, deleteResolver } =
    React.useMemo(() => {
      switch (kind) {
        case RepositoryKind.Github:
          return {
            updateMutation: updateGithubRepositoryProviderMutation,
            deleteMutation: deleteGithubRepositoryProviderMutation,
            updateResolver: (
              res?: QueryResponseData<
                typeof updateGithubRepositoryProviderMutation
              >
            ) => res?.updateGithubRepositoryProvider,
            deleteResolver: (
              res?: QueryResponseData<
                typeof deleteGithubRepositoryProviderMutation
              >
            ) => res?.deleteGithubRepositoryProvider
          }
        case RepositoryKind.GithubSelfHosted:
          return {
            updateMutation: updateGithubSelfHostedRepositoryProviderMutation,
            deleteMutation: deleteGithubSelfHostedRepositoryProviderMutation,
            updateResolver: (
              res?: QueryResponseData<
                typeof updateGithubSelfHostedRepositoryProviderMutation
              >
            ) => res?.updateGithubSelfHostedRepositoryProvider,
            deleteResolver: (
              res?: QueryResponseData<
                typeof deleteGithubSelfHostedRepositoryProviderMutation
              >
            ) => res?.deleteGithubSelfHostedRepositoryProvider
          }
        case RepositoryKind.Gitlab:
          return {
            updateMutation: updateGitlabRepositoryProviderMutation,
            deleteMutation: deleteGitlabRepositoryProviderMutation,
            updateResolver: (
              res?: QueryResponseData<
                typeof updateGitlabRepositoryProviderMutation
              >
            ) => res?.updateGitlabRepositoryProvider,
            deleteResolver: (
              res?: QueryResponseData<
                typeof deleteGitlabRepositoryProviderMutation
              >
            ) => res?.deleteGitlabRepositoryProvider
          }
        case RepositoryKind.GitlabSelfHosted:
          return {
            updateMutation: updateGitlabSelfHostedRepositoryProviderMutation,
            deleteMutation: deleteGitlabSelfHostedRepositoryProviderMutation,
            updateResolver: (
              res?: QueryResponseData<
                typeof updateGitlabSelfHostedRepositoryProviderMutation
              >
            ) => res?.updateGitlabSelfHostedRepositoryProvider,
            deleteResolver: (
              res?: QueryResponseData<
                typeof deleteGitlabSelfHostedRepositoryProviderMutation
              >
            ) => res?.deleteGitlabSelfHostedRepositoryProvider
          }
        default:
          return {}
      }
    }, [kind]) as {
      updateMutation: TypedDocumentNode<any, any>
      deleteMutation: TypedDocumentNode<any, any>
      updateResolver: (data?: Record<string, boolean>) => boolean | undefined
      deleteResolver: (data?: Record<string, boolean>) => boolean | undefined
    }

  const deleteRepositoryProvider = useMutation(deleteMutation)

  const updateRepositoryProvider = useMutation(updateMutation, {
    form
  })

  const onSubmit = async (values: CreateRepositoryProviderFormValues) => {
    const res = await updateRepositoryProvider({
      input: {
        id,
        ...values
      }
    })
    if (updateResolver?.(res?.data)) {
      toast.success('Updated GitHub repository provider successfully')
      form?.reset(form?.getValues())
      onSuccess?.()
      onUpdate?.()
    }
  }

  const handleDeleteRepositoryProvider = async () => {
    const res = await deleteRepositoryProvider({ id })
    if (deleteResolver?.(res?.data)) {
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
