'use client'

import React from 'react'
import { useRouter } from 'next/navigation'
import { UseFormReturn } from 'react-hook-form'

import { graphql } from '@/lib/gql/generates'
import { RepositoryKind } from '@/lib/gql/generates/graphql'
import { useMutation } from '@/lib/tabby/gql'

import {
  CommonProviderForm,
  CreateRepositoryProviderFormValues,
  UpdateRepositoryProviderFormValues,
  useRepositoryProviderForm
} from '../../components/common-provider-form'
import { useRepositoryKind } from '../../hooks/use-repository-kind'

const createGithubRepositoryProvider = graphql(/* GraphQL */ `
  mutation CreateGithubRepositoryProvider(
    $input: CreateRepositoryProviderInput!
  ) {
    createGithubRepositoryProvider(input: $input)
  }
`)

const createGitlabRepositoryProvider = graphql(/* GraphQL */ `
  mutation CreateGitlabRepositoryProvider(
    $input: CreateRepositoryProviderInput!
  ) {
    createGitlabRepositoryProvider(input: $input)
  }
`)

export const NewProvider = () => {
  const kind = useRepositoryKind()
  const router = useRouter()
  const form = useRepositoryProviderForm(true)
  // for github
  const createGithubRepositoryProviderMutation = useMutation(
    createGithubRepositoryProvider,
    {
      onCompleted(data) {
        if (data?.createGithubRepositoryProvider) {
          router.back()
        }
      },
      form
    }
  )

  // for gitlab
  const createGitlabRepositoryProviderMutation = useMutation(
    createGitlabRepositoryProvider,
    {
      onCompleted(data) {
        if (data?.createGitlabRepositoryProvider) {
          router.back()
        }
      },
      form
    }
  )

  const handleSubmit = async (values: CreateRepositoryProviderFormValues) => {
    if (kind === RepositoryKind.Github) {
      return createGithubRepositoryProviderMutation({
        input: values
      })
    }
    if (kind === RepositoryKind.Gitlab) {
      return createGitlabRepositoryProviderMutation({
        input: values
      })
    }
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
