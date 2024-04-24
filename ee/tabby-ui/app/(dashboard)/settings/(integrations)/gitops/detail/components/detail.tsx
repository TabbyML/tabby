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
import { IconChevronLeft, IconGitHub, IconTrash } from '@/components/ui/icons'
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
  const [githubRepositories, fetchingRepositories] =
    useAllProvidedRepositories(id)

  const [open, setOpen] = React.useState(false)

  const onCreated = () => {
    toast.success('Created successfully')
    setOpen(false)
  }

  // const onDeleteRepo = () => {
  // }

  const onDeleteProvider = () => {
    router.replace('/settings/gitops')
  }

  const unlinkedRepos = useMemo(() => {
    return githubRepositories?.filter(item => !item.node.active)
  }, [githubRepositories])

  const linkedRepos = useMemo(() => {
    return githubRepositories?.filter(item => item.node.active)
  }, [githubRepositories])

  if (!id || (!!id && !fetching && !provider)) {
    return (
      <div className="flex h-[250px] w-full items-center justify-center rounded-lg border">
        Provider not found
      </div>
    )
  }

  return (
    <LoadingWrapper loading={fetching}>
      <CardHeader className="pl-0 pt-0">
        <CardTitle className="flex items-center justify-between">
          <div
            onClick={() => router.back()}
            className="-ml-1 flex cursor-pointer items-center transition-opacity hover:opacity-60"
          >
            <IconChevronLeft className="mr-1 h-6 w-6" />
            <span>
              Provider information
            </span>
          </div>
          <div className="flex items-center gap-2 text-base">
            <IconGitHub className="h-6 w-6" />
            GitHub.com
            <div className="ml-1">
              {provider?.connected ? (
                <Badge variant="successful">Connected</Badge>
              ) : (
                <Badge variant="destructive">Not Connected</Badge>
              )}
            </div>
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent className="pl-0">
        <LoadingWrapper loading={fetching} fallback={<ListSkeleton />}>
          <UpdateProviderForm
            defaultValues={provider}
            onDelete={onDeleteProvider}
            onBack={() => {
              router.push('/settings/gitops')
            }}
            id={id}
          />
        </LoadingWrapper>
      </CardContent>

      <CardHeader className="mt-8 pl-0 pt-0">
        <CardTitle className="flex items-center justify-between">
          <span>Linked repositories</span>
          <Dialog open={open} onOpenChange={setOpen}>
            <DialogContent>
              <DialogHeader className="gap-3">
                <DialogTitle>New repository link</DialogTitle>
                <DialogDescription>
                  Add a new repository link to this provider.
                </DialogDescription>
              </DialogHeader>
              <LinkRepositoryForm
                onCancel={() => setOpen(false)}
                onCreated={onCreated}
                repositories={unlinkedRepos}
                fetchingRepositories={fetchingRepositories}
              />
            </DialogContent>
            <DialogTrigger asChild>
              <Button>New repository link</Button>
            </DialogTrigger>
          </Dialog>
        </CardTitle>
      </CardHeader>
      <CardContent className="pl-0">
        <LoadingWrapper loading={fetchingRepositories}>
          <LinkedRepoTable data={linkedRepos} />
        </LoadingWrapper>
      </CardContent>
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

  return (
    <Table className="mt-4">
      <TableHeader>
        <TableRow>
          <TableHead className="w-[25%]">Repository name</TableHead>
          <TableHead className="w-[45%]">Git url</TableHead>
          <TableHead></TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {data?.length ? (
          <>
            {data?.map(x => {
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
              No linked repositories yet.
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
  }, [queryVariables, fetching])

  return [data?.githubRepositories?.edges ?? [], !isAllLoaded]
}

export default DetailPage
