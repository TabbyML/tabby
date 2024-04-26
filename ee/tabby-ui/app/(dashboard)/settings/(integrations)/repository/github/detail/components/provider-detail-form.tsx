'use client'

import React from 'react'
import { UseFormReturn } from 'react-hook-form'
import { toast } from 'sonner'
import * as z from 'zod'

import { graphql } from '@/lib/gql/generates'
import { useMutation } from '@/lib/tabby/gql'

import {
  GithubProviderForm,
  UpdateGithubProviderFormValues,
  updateGithubProviderSchema
} from '../../components/github-form'

const deleteGithubRepositoryProviderMutation = graphql(/* GraphQL */ `
  mutation DeleteRepositoryProvider($id: ID!) {
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

type FormValues = z.infer<typeof updateGithubProviderSchema>

interface UpdateProviderFormProps {
  id: string
  defaultValues?: Partial<FormValues>
  onSuccess?: () => void
  onDelete: () => void
}

export const UpdateProviderForm: React.FC<UpdateProviderFormProps> = ({
  defaultValues,
  onSuccess,
  onDelete,
  id
}) => {
  const formRef = React.useRef<{
    form: UseFormReturn<UpdateGithubProviderFormValues>
  }>(null)
  const form = formRef.current?.form

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

  const onSubmit = async (values: FormValues) => {
    await updateGithubRepositoryProvider({
      input: {
        id,
        ...values
      }
    })
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
    <GithubProviderForm
      ref={formRef}
      defaultValues={defaultValues}
      onSubmit={onSubmit}
      onDelete={handleDeleteRepositoryProvider}
      deletable
      cancleable={false}
    />
  )
}
