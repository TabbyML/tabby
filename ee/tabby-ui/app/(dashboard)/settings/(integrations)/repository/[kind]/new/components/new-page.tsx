'use client'

import React from 'react'
import { useRouter } from 'next/navigation'
import { UseFormReturn } from 'react-hook-form'

import { graphql } from '@/lib/gql/generates'
import { RepositoryKind } from '@/lib/gql/generates/graphql'
import { QueryResponseData, useMutation } from '@/lib/tabby/gql'

import {
  CommonProviderForm,
  CreateRepositoryProviderFormValues,
  UpdateRepositoryProviderFormValues,
  useRepositoryProviderForm
} from '../../components/common-provider-form'
import { useRepositoryKind } from '../../hooks/use-repository-kind'
import { TypedDocumentNode } from 'urql'

const createGithubRepositoryProvider = graphql(/* GraphQL */ `
  mutation CreateGithubRepositoryProvider(
    $input: CreateRepositoryProviderInput!
  ) {
    createGithubRepositoryProvider(input: $input)
  }
`)

const createGithubSelfHostedRepositoryProvider = graphql(/* GraphQL */ `
  mutation CreateGithubSelfHostedRepositoryProvider(
    $input: CreateSelfHostedRepositoryProviderInput!
  ) {
    createGithubSelfHostedRepositoryProvider(input: $input)
  }
`)

const createGitlabRepositoryProvider = graphql(/* GraphQL */ `
  mutation CreateGitlabRepositoryProvider(
    $input: CreateRepositoryProviderInput!
  ) {
    createGitlabRepositoryProvider(input: $input)
  }
`)

const createGitlabSelfHostedRepositoryProvider = graphql(/* GraphQL */ `
  mutation CreateGitlabSelfHostedRepositoryProvider(
    $input: CreateSelfHostedRepositoryProviderInput!
  ) {
    createGitlabSelfHostedRepositoryProvider(input: $input)
  }
`)

export const NewProvider = () => {
  const kind = useRepositoryKind()
  const router = useRouter()
  const form = useRepositoryProviderForm(true)

  const { mutation, resolver } = React.useMemo(() => {
    switch (kind) {
      case RepositoryKind.Github:
        return {
          mutation: createGithubRepositoryProvider,
          resolver: (
            res?: QueryResponseData<typeof createGithubRepositoryProvider>
          ) => res?.createGithubRepositoryProvider
        }
      case RepositoryKind.GithubSelfHosted:
        return {
          mutation: createGithubSelfHostedRepositoryProvider,
          resolver: (
            res?: QueryResponseData<
              typeof createGithubSelfHostedRepositoryProvider
            >
          ) => res?.createGithubSelfHostedRepositoryProvider
        }
      case RepositoryKind.Gitlab:
        return {
          mutation: createGitlabRepositoryProvider,
          resolver: (
            res?: QueryResponseData<typeof createGitlabRepositoryProvider>
          ) => res?.createGitlabRepositoryProvider
        }
      case RepositoryKind.GitlabSelfHosted:
        return {
          mutation: createGitlabSelfHostedRepositoryProvider,
          resolver: (
            res?: QueryResponseData<
              typeof createGitlabSelfHostedRepositoryProvider
            >
          ) => res?.createGitlabSelfHostedRepositoryProvider
        }
      default:
        return {
          mutation: createGithubRepositoryProvider,
          resolver: (
            res?: QueryResponseData<typeof createGithubRepositoryProvider>
          ) => res?.createGithubRepositoryProvider
        }
    }
  }, [kind]) as {
    mutation: TypedDocumentNode<any, any>
    resolver: (
      res?: QueryResponseData<TypedDocumentNode<any, any>>
    ) => string | undefined
  }

  const createRepositoryProviderMutation = useMutation(mutation, {
    onCompleted(data) {
      if (resolver(data)) {
        router.back()
      }
    },
    form
  })

  const handleSubmit = async (values: CreateRepositoryProviderFormValues) => {
    return createRepositoryProviderMutation({
      input: values
    })
  }

  return (
    <div className="ml-4">
      <CommonProviderForm
        isNew
        form={form as UseFormReturn<UpdateRepositoryProviderFormValues>}
        onSubmit={handleSubmit}
      />
    </div>
  )
}
