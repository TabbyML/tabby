'use client'

import React from 'react'
import { useRouter } from 'next/navigation'
import { UseFormReturn } from 'react-hook-form'

import { graphql } from '@/lib/gql/generates'
import { useMutation } from '@/lib/tabby/gql'

import {
  CommonProviderForm,
  CreateRepositoryProviderFormValues,
  UpdateRepositoryProviderFormValues,
  useRepositoryProviderForm
} from '../../components/common-provider-form'
import { useIntegrationKind } from '../../hooks/use-repository-kind'

const createIntegration = graphql(/* GraphQL */ `
  mutation CreateIntegration($input: CreateIntegrationInput!) {
    createIntegration(input: $input)
  }
`)

export const NewProvider = () => {
  const kind = useIntegrationKind()
  const router = useRouter()
  const form = useRepositoryProviderForm(true, kind)

  const createRepositoryProviderMutation = useMutation(createIntegration, {
    onCompleted(data) {
      if (data?.createIntegration) {
        router.back()
      }
    },
    form
  })

  const handleSubmit = async (values: CreateRepositoryProviderFormValues) => {
    return createRepositoryProviderMutation({
      input: {
        ...values,
        kind
      }
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
