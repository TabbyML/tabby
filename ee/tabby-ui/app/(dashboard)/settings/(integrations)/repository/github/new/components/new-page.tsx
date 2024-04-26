'use client'

import React from 'react'
import { useRouter } from 'next/navigation'
import { omit } from 'lodash-es'
import { UseFormReturn } from 'react-hook-form'
import { toast } from 'sonner'

import { graphql } from '@/lib/gql/generates'
import { useMutation } from '@/lib/tabby/gql'

import {
  CreateGithubProviderFormValues,
  GithubProviderForm,
  UpdateGithubProviderFormValues
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
  const formRef = React.useRef<{
    form: UseFormReturn<UpdateGithubProviderFormValues>
  }>(null)
  const createGithubRepositoryProviderMutation = useMutation(
    createGithubRepositoryProvider,
    {
      onCompleted(data) {
        if (data?.createGithubRepositoryProvider) {
          router.back()
        }
      },
      onError(err) {
        toast.error(err?.message)
      },
      form: formRef.current
    }
  )

  const handleSubmit = async (values: CreateGithubProviderFormValues) => {
    return createGithubRepositoryProviderMutation({
      input: omit(values, 'provider')
    })
  }

  return (
    <div className="ml-4">
      <GithubProviderForm isNew ref={formRef} onSubmit={handleSubmit} />
    </div>
  )
}
