'use client'

import React, { useEffect, useMemo, useState } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { toast } from 'sonner'
import { useQuery } from 'urql'

import { DEFAULT_PAGE_SIZE } from '@/lib/constants'
import { QueryResponseData, QueryVariables, useMutation } from '@/lib/tabby/gql'
import {
  listGithubRepositories,
  listGithubRepositoryProviders
} from '@/lib/tabby/query'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger
} from '@/components/ui/dialog'
import { IconChevronLeft, IconGitHub, IconPlus, IconTrash } from '@/components/ui/icons'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow
} from '@/components/ui/table'
import LoadingWrapper from '@/components/loading-wrapper'
import { ListSkeleton } from '@/components/skeleton'

import { updateGithubProvidedRepositoryActiveMutation } from '../query'
import LinkRepositoryForm from './new-repository-form'
import { UpdateProviderForm } from './provider-detail-form'
import { Separator } from '@/components/ui/separator'

type GithubRepositories = QueryResponseData<
  typeof listGithubRepositories
>['githubRepositories']['edges']

const DetailPage: React.FC = () => {
  const searchParams = useSearchParams()
  const router = useRouter()
  const id = searchParams.get('id')?.toString() ?? ''
  const [{ data, fetching }] = useQuery({
    query: listGithubRepositoryProviders,
    variables: { ids: [id] },
    pause: !id
  })
  const provider = data?.githubRepositoryProviders?.edges?.[0]?.node
  const [githubRepositories, isGithubRepositoriesLoading] =
    useAllProvidedRepositories(id)
  console.log(githubRepositories)

  const onDeleteProvider = () => {
    router.back()
  }

  if (!id || (!!id && !fetching && !provider)) {
    return (
      <div className="flex h-[250px] w-full items-center justify-center rounded-lg border">
        Provider not found
      </div>
    )
  }

  return (
    <LoadingWrapper loading={fetching}>
      <CardTitle className="flex items-center justify-between">
        <div className="-ml-1 flex items-center">
          <Button
            onClick={() => router.back()}
            variant={'ghost'}
            className="h-6 px-1"
          >
            <IconChevronLeft className="h-5 w-5" />
          </Button>
          <span className="ml-2">{provider?.displayName}</span>
        </div>
        <div className="flex items-center gap-2 text-base">
          <div className="ml-1">
            {provider?.connected ? (
              <Badge variant="successful">Connected</Badge>
            ) : (
              <Badge variant="destructive">Not Connected</Badge>
            )}
          </div>
        </div>
      </CardTitle>
      <CardContent className="mt-8">
        <LoadingWrapper loading={fetching} fallback={<ListSkeleton />}>
          <UpdateProviderForm
            defaultValues={provider}
            onDelete={onDeleteProvider}
            id={id}
          />
        </LoadingWrapper>
      </CardContent>

      <div className="p-4">
        <LoadingWrapper loading={isGithubRepositoriesLoading}>
          <LinkedRepoTable data={githubRepositories} />
        </LoadingWrapper>
      </div>
    </LoadingWrapper>
  )
}

const LinkedRepoTable: React.FC<{
  data: GithubRepositories | undefined
  onDelete?: () => void
}> = ({ data, onDelete }) => {
  // const [isDeleting, setIsDeleting] = useState(false)
  const updateGithubProvidedRepositoryActive = useMutation(
    updateGithubProvidedRepositoryActiveMutation,
    {
      onCompleted(data) {
        if (data?.updateGithubProvidedRepositoryActive) {
          onDelete?.()
        }
        // setIsDeleting(false)
      },
      onError() {
        // setIsDeleting(false)
      }
    }
  )

  const handleDelete = async (id: string) => {
    // setIsDeleting(true)
    updateGithubProvidedRepositoryActive({
      id,
      active: false
    })
  }

  const [open, setOpen] = React.useState(false)

  const onCreated = () => {
    setOpen(false)
  }

  const linkedRepos = useMemo(() => {
    return data?.filter(item => item.node.active)
  }, [data])

  const unlinkedRepos = useMemo(() => {
    return data?.filter(item => !item.node.active)
  }, [data])

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead className="w-[25%]">Name</TableHead>
          <TableHead className="w-[45%]">URL</TableHead>
          <TableHead className='text-right'>
            <Dialog open={open} onOpenChange={setOpen}>
              <DialogContent>
                <DialogHeader className="gap-3">
                  <DialogTitle>Add new repository</DialogTitle>
                  <DialogDescription>
                    Add new GitHub repository from this provider
                  </DialogDescription>
                </DialogHeader>
                <LinkRepositoryForm
                  onCancel={() => setOpen(false)}
                  onCreated={onCreated}
                  repositories={unlinkedRepos}
                />
              </DialogContent>
              <DialogTrigger asChild>
                <Button variant="ghost" size="icon"><IconPlus /></Button>
              </DialogTrigger>
            </Dialog>

          </TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {linkedRepos?.length ? (
          <>
            {linkedRepos?.map(x => {
              return (
                <TableRow key={x.node.id}>
                  <TableCell>{x.node.name}</TableCell>
                  <TableCell>{x.node.gitUrl}</TableCell>
                  <TableCell className="flex justify-end">
                    <div className="flex gap-1">
                      <Button
                        size="icon"
                        variant="hover-destructive"
                        onClick={e => handleDelete(x.node.id)}
                      >
                        <IconTrash />
                      </Button>
                    </div>
                  </TableCell>
                </TableRow>
              )
            })}
          </>
        ) : (
          <TableRow>
            <TableCell colSpan={3} className="h-[100px] text-center">
              No repositories yet.
            </TableCell>
          </TableRow>
        )}
      </TableBody>
    </Table>
  )
}

export function useAllProvidedRepositories(
  id: string
): [GithubRepositories, boolean] {
  const [queryVariables, setQueryVariables] = useState<
    QueryVariables<typeof listGithubRepositories>
  >({ providerIds: [id], first: DEFAULT_PAGE_SIZE })

  const [isAllLoaded, setIsAllLoaded] = useState(!id)

  const [{ data, fetching }] = useQuery({
    query: listGithubRepositories,
    variables: queryVariables,
    pause: !id
  })

  useEffect(() => {
    if (isAllLoaded) return
    if (!fetching && data) {
      const pageInfo = data?.githubRepositories?.pageInfo

      if (pageInfo?.hasNextPage) {
        setQueryVariables({
          providerIds: [id],
          first: DEFAULT_PAGE_SIZE,
          after: pageInfo.endCursor
        })
      } else {
        setIsAllLoaded(true)
      }
    }
  }, [fetching, data])

  return [data?.githubRepositories?.edges ?? [], !isAllLoaded]
}

export default DetailPage
