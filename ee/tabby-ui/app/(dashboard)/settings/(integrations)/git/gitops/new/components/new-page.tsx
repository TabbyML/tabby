'use client'

import { useRouter } from 'next/navigation'
import { createRequest } from '@urql/core'
import { toast } from 'sonner'

import { graphql } from '@/lib/gql/generates'
import { client, useMutation } from '@/lib/tabby/gql'
import { listGithubRepositoryProviders } from '@/lib/tabby/query'
import { Button } from '@/components/ui/button'
import { CardHeader, CardTitle } from '@/components/ui/card'
import { IconChevronLeft } from '@/components/ui/icons'

import { GitProviderForm } from '../../components/git-provider-form'

const createGithubRepositoryProvider = graphql(/* GraphQL */ `
  mutation CreateGithubRepositoryProvider(
    $input: CreateGithubRepositoryProviderInput!
  ) {
    createGithubRepositoryProvider(input: $input)
  }
`)

export const NewProvider = () => {
  const router = useRouter()

  const getProvider = (id: string) => {
    const queryProvider = client.createRequestOperation(
      'query',
      createRequest(listGithubRepositoryProviders, { ids: [id] })
    )
    return client.executeQuery(queryProvider)
  }

  // const createGithubRepositoryProviderMutation = useMutation(
  //   createGithubRepositoryProvider,
  //   {
  //     onCompleted(data) {
  //       if (data?.createGithubRepositoryProvider) {
  //       }
  //     },
  //     onError(err) {
  //       toast.error(err?.message)
  //     },
  //     form
  //   }
  // )

  const handleSubmit = async () => {
    // createGithubRepositoryProviderMutation({
    //   input: values
    // })
  }

  return (
    <>
      <CardHeader className="pl-0">
        <CardTitle>
          <div className="-ml-1 flex items-center">
            <Button
              onClick={() => router.back()}
              variant={'ghost'}
              className="px-1"
            >
              <IconChevronLeft className="w-6 h-6" />
            </Button>
            <span className="ml-2">Create Git Provider</span>
          </div>
        </CardTitle>
      </CardHeader>
      <GitProviderForm isNew defaultValues={{ provider: 'github' }} />
    </>
  )
}
