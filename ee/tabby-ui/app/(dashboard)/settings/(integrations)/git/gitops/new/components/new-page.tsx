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
import { CardTitle } from '@/components/ui/card'
import { FormMessage } from '@/components/ui/form'
import { IconChevronLeft, IconSpinner } from '@/components/ui/icons'

import {
  CreateGitProviderFormValues,
  GitProviderForm,
  UpdateGitProviderFormValues
} from '../../components/git-provider-form'

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
    form: UseFormReturn<UpdateGitProviderFormValues>
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
          toast.success('Provider created successfully')
          router.replace('/settings/git/gitops')
        }
      },
      onError(err) {
        toast.error(err?.message)
      },
      form: formRef.current
    }
  )

  const handleSubmit = async (values: CreateGitProviderFormValues) => {
    return createGithubRepositoryProviderMutation({
      input: omit(values, 'provider')
    })
  }

  return (
    <>
      <CardTitle className="py-6">
        <div className="-ml-1 flex items-center">
          <Button
            onClick={() => router.back()}
            variant={'ghost'}
            className="h-6 px-1"
          >
            <IconChevronLeft className="h-5 w-5" />
          </Button>
          <span className="ml-2">Create Git Provider</span>
        </div>
      </CardTitle>
      <GitProviderForm
        isNew
        ref={formRef}
        defaultValues={{ provider: 'github' }}
        footer={
          <div className="flex items-center justify-between">
            <div>
              <FormMessage />
            </div>
            <div>
              <Button type="submit" disabled={isSubmitting}>
                {isSubmitting && <IconSpinner className="mr-2" />}
                Create
              </Button>
            </div>
          </div>
        }
        onSubmit={handleSubmit}
      />
    </>
  )
}
