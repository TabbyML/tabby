'use client'

import React from 'react'
import { useRouter } from 'next/navigation'
import { createRequest } from '@urql/core'
import { omit } from 'lodash-es'
import { UseFormReturn } from 'react-hook-form'
import { toast } from 'sonner'

import { graphql } from '@/lib/gql/generates'
import { client, useMutation } from '@/lib/tabby/gql'
import { listGithubRepositoryProviders } from '@/lib/tabby/query'
import { Button } from '@/components/ui/button'
import { FormMessage } from '@/components/ui/form'
import { IconSpinner } from '@/components/ui/icons'

import {
  CreateGithubProviderFormValues,
  GithubProviderForm,
  UpdateGithubProviderFormValues
} from '../../components/github-form'

const createGithubRepositoryProvider = graphql(/* GraphQL */ `
  mutation CreateGithubRepositoryProvider(
    $input: CreateGithubRepositoryProviderInput!
  ) {
    createGithubRepositoryProvider(input: $input)
  }
`)

export const NewProvider = () => {
  const router = useRouter()
  const formRef = React.useRef<{
    form: UseFormReturn<UpdateGithubProviderFormValues>
  }>(null)
  const isSubmitting = formRef.current?.form?.formState?.isSubmitting

  const getProvider = (id: string) => {
    const queryProvider = client.createRequestOperation(
      'query',
      createRequest(listGithubRepositoryProviders, { ids: [id] })
    )
    return client.executeQuery(queryProvider)
  }

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
      <GithubProviderForm
        isNew
        ref={formRef}
        defaultValues={{ provider: 'github' }}
        footer={
          <div className="flex items-center justify-between">
            <div>
              <FormMessage />
            </div>
            <div className="flex gap-2">
              <Button
                type="button"
                variant="ghost"
                disabled={isSubmitting}
                onClick={() => router.back()}
              >
                Cancel
              </Button>
              <Button type="submit" disabled={isSubmitting}>
                {isSubmitting && <IconSpinner className="mr-2" />}
                Create
              </Button>
            </div>
          </div>
        }
        onSubmit={handleSubmit}
      />
    </div>
  )
}
