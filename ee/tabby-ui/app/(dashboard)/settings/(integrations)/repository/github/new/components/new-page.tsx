'use client'

import React from 'react'
import { useRouter } from 'next/navigation'

import { graphql } from '@/lib/gql/generates'
import { useMutation } from '@/lib/tabby/gql'

import {
  GithubProviderForm,
  RepositoryProviderFormValues,
  useRepositoryProviderForm
} from '../../components/github-form'

const createGithubRepositoryProvider = graphql(/* GraphQL */ `
  mutation CreateGithubRepositoryProvider(
    $input: CreateRepositoryProviderInput!
  ) {
    createGithubRepositoryProvider(input: $input)
  }
`)

export const NewProvider = () => {
  const router = useRouter()
  const form = useRepositoryProviderForm()
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

  const handleSubmit = async (values: RepositoryProviderFormValues) => {
    return createGithubRepositoryProviderMutation({
      input: values
    })
  }

  return (
    <div className="ml-4">
      <GithubProviderForm isNew form={form} onSubmit={handleSubmit} />
    </div>
  )
}
