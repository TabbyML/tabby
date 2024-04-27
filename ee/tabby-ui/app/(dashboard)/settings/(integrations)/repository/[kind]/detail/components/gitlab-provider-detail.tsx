'use client'

import React, { useEffect, useMemo, useState } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { toast } from 'sonner'
import { useQuery } from 'urql'

import { DEFAULT_PAGE_SIZE } from '@/lib/constants'
import {
  RepositoryKind,
  RepositoryProviderStatus
} from '@/lib/gql/generates/graphql'
import { QueryResponseData, QueryVariables, useMutation } from '@/lib/tabby/gql'
import {
  listGitlabRepositories,
  listGitlabRepositoryProviders
} from '@/lib/tabby/query'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { CardContent, CardTitle } from '@/components/ui/card'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger
} from '@/components/ui/dialog'
import { IconChevronLeft, IconPlus, IconTrash } from '@/components/ui/icons'
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

import { updateGitlabProvidedRepositoryActiveMutation } from '../query'
import LinkRepositoryForm from './add-repository-form'
import { UpdateProviderForm } from './update-gitlab-provider-form'

type GitlabRepositories = QueryResponseData<
  typeof listGitlabRepositories
>['gitlabRepositories']['edges']

const GitlabProviderDetail: React.FC = () => {
  const searchParams = useSearchParams()
  const router = useRouter()
  const id = searchParams.get('id')?.toString() ?? ''
  const [{ data, fetching }] = useQuery({
    query: listGitlabRepositoryProviders,
    variables: { ids: [id] },
    pause: !id
  })
  const provider = data?.gitlabRepositoryProviders?.edges?.[0]?.node
  const [gitlabRepositories, isGitlabRepositoriesLoading] =
    useAllProvidedRepositories(id)

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
      <CardTitle className="flex items-center gap-4">
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
            {provider && toStatusBadge(provider.status)}
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
        <LoadingWrapper loading={isGitlabRepositoriesLoading}>
          <LinkedRepoTable
            data={gitlabRepositories}
            providerStatus={provider?.status}
          />
        </LoadingWrapper>
      </div>
    </LoadingWrapper>
  )
}

function toStatusBadge(status: RepositoryProviderStatus) {
  switch (status) {
    case RepositoryProviderStatus.Ready:
      return <Badge variant="successful">Ready</Badge>
    case RepositoryProviderStatus.Error:
      return <Badge variant="destructive">Error</Badge>
    case RepositoryProviderStatus.Error:
      return <Badge>Pending</Badge>
  }
}

const LinkedRepoTable: React.FC<{
  data: GitlabRepositories | undefined
  providerStatus: RepositoryProviderStatus | undefined
  onDelete?: () => void
}> = ({ data, onDelete, providerStatus }) => {
  const updateGitlabProvidedRepositoryActive = useMutation(
    updateGitlabProvidedRepositoryActiveMutation,
    {
      onCompleted(data) {
        if (data?.updateGitlabProvidedRepositoryActive) {
          onDelete?.()
        }
      },
      onError(error) {
        toast.error(error.message || 'Failed to delete')
      }
    }
  )

  const handleDelete = async (id: string) => {
    updateGitlabProvidedRepositoryActive({
      id,
      active: false
    })
  }

  const [open, setOpen] = React.useState(false)

  const onCreated = () => {
    setOpen(false)
  }

  const activeRepos = useMemo(() => {
    return data?.filter(item => item.node.active)
  }, [data])

  const inactiveRepos = useMemo(() => {
    return data?.filter(item => !item.node.active)
  }, [data])

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead className="w-[25%]">Name</TableHead>
          <TableHead className="w-[45%]">URL</TableHead>
          <TableHead className="text-right">
            <Dialog open={open} onOpenChange={setOpen}>
              <DialogContent>
                <DialogHeader className="gap-3">
                  <DialogTitle>Add new repository</DialogTitle>
                  <DialogDescription>
                    Add new GitLab repository from this provider
                  </DialogDescription>
                </DialogHeader>
                <LinkRepositoryForm
                  onCancel={() => setOpen(false)}
                  onCreated={onCreated}
                  repositories={inactiveRepos}
                  kind={RepositoryKind.Gitlab}
                  providerStatus={providerStatus}
                />
              </DialogContent>
              <DialogTrigger asChild>
                <Button variant="ghost" size="icon">
                  <IconPlus />
                </Button>
              </DialogTrigger>
            </Dialog>
          </TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {activeRepos?.length ? (
          <>
            {activeRepos?.map(x => {
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

function useAllProvidedRepositories(id: string): [GitlabRepositories, boolean] {
  const [queryVariables, setQueryVariables] = useState<
    QueryVariables<typeof listGitlabRepositories>
  >({ providerIds: [id], first: DEFAULT_PAGE_SIZE })

  const [isAllLoaded, setIsAllLoaded] = useState(!id)

  const [{ data, fetching }] = useQuery({
    query: listGitlabRepositories,
    variables: queryVariables,
    pause: !id
  })

  useEffect(() => {
    if (isAllLoaded) return
    if (!fetching && data) {
      const pageInfo = data?.gitlabRepositories?.pageInfo

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

  return [data?.gitlabRepositories?.edges ?? [], !isAllLoaded]
}

export default GitlabProviderDetail
