'use client'

import React, { useEffect, useState } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { toast } from 'sonner'
import { useQuery } from 'urql'

import { DEFAULT_PAGE_SIZE } from '@/lib/constants'
import {
  ListGitlabRepositoriesQuery,
  RepositoryKind,
  RepositoryProviderStatus
} from '@/lib/gql/generates/graphql'
import { useDebounceCallback } from '@/lib/hooks/use-debounce'
import {
  client,
  QueryResponseData,
  QueryVariables,
  useMutation
} from '@/lib/tabby/gql'
import {
  listGitlabRepositories,
  listGitlabRepositoryProviders
} from '@/lib/tabby/query'
import { Badge } from '@/components/ui/badge'
import { Button, buttonVariants } from '@/components/ui/button'
import { CardContent, CardTitle } from '@/components/ui/card'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger
} from '@/components/ui/dialog'
import {
  IconChevronLeft,
  IconChevronRight,
  IconPlus,
  IconSpinner,
  IconTrash
} from '@/components/ui/icons'
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
import AddRepositoryForm from './add-repository-form'
import { UpdateProviderForm } from './update-gitlab-provider-form'

type GitlabRepositories = QueryResponseData<
  typeof listGitlabRepositories
>['gitlabRepositories']['edges']

const GitlabProviderDetail: React.FC = () => {
  const searchParams = useSearchParams()
  const router = useRouter()
  const id = searchParams.get('id')?.toString() ?? ''
  const [{ data, fetching }, reexecuteQuery] = useQuery({
    query: listGitlabRepositoryProviders,
    variables: { ids: [id] },
    pause: !id
  })
  const provider = data?.gitlabRepositoryProviders?.edges?.[0]?.node

  const onDeleteProvider = () => {
    router.back()
  }

  const onUpdateProvider = () => {
    reexecuteQuery()
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
        <div className="-ml-2.5 flex items-center">
          <Button
            onClick={() => router.back()}
            variant={'ghost'}
            className="h-6 px-1"
          >
            <IconChevronLeft className="h-5 w-5" />
          </Button>
          <span className="ml-1">{provider?.displayName}</span>
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
            onUpdate={onUpdateProvider}
            id={id}
          />
        </LoadingWrapper>
      </CardContent>

      <div className="p-4">
        <ActiveRepoTable providerId={id} providerStatus={provider?.status} />
      </div>
    </LoadingWrapper>
  )
}

function toStatusBadge(status: RepositoryProviderStatus) {
  switch (status) {
    case RepositoryProviderStatus.Ready:
      return <Badge variant="successful">Ready</Badge>
    case RepositoryProviderStatus.Failed:
      return <Badge variant="destructive">Error</Badge>
    case RepositoryProviderStatus.Pending:
      return <Badge>Pending</Badge>
  }
}

const ActiveRepoTable: React.FC<{
  providerId: string
  providerStatus: RepositoryProviderStatus | undefined
  onDelete?: () => void
}> = ({ onDelete, providerStatus, providerId }) => {
  const [page, setPage] = React.useState(1)
  const {
    repositories: inactiveRepositories,
    setRepositories: setInactiveRepositories,
    isAllLoaded: isInactiveRepositoriesLoaded
  } = useAllInactiveRepositories(providerId)

  const fetchRepositories = (
    variables: QueryVariables<typeof listGitlabRepositories>
  ) => {
    return client.query(listGitlabRepositories, variables).toPromise()
  }

  const fetchRepositoriesSequentially = async (
    page: number,
    cursor?: string
  ): Promise<ListGitlabRepositoriesQuery | undefined> => {
    const res = await fetchRepositories({
      providerIds: [providerId],
      first: DEFAULT_PAGE_SIZE,
      after: cursor,
      active: true
    })
    const _pageInfo = res?.data?.gitlabRepositories?.pageInfo
    if (page - 1 > 0 && _pageInfo?.hasNextPage && _pageInfo?.endCursor) {
      return fetchRepositoriesSequentially(page - 1, _pageInfo.endCursor)
    } else {
      return res?.data
    }
  }

  const [activeRepositoriesResult, setActiveRepositoriesResult] =
    React.useState<QueryResponseData<typeof listGitlabRepositories>>()
  const [fetching, setFetching] = React.useState(true)
  const [recentlyActivatedRepositories, setRecentlyActivatedRepositories] =
    React.useState<GitlabRepositories>([])
  const activeRepos = activeRepositoriesResult?.gitlabRepositories?.edges
  const pageInfo = activeRepositoriesResult?.gitlabRepositories?.pageInfo

  const updateGitlabProvidedRepositoryActive = useMutation(
    updateGitlabProvidedRepositoryActiveMutation,
    {
      onError(error) {
        toast.error(error.message || 'Failed to delete')
      }
    }
  )

  const handleDelete = async (
    repo: GitlabRepositories[0],
    isLastItem?: boolean
  ) => {
    updateGitlabProvidedRepositoryActive({
      id: repo.node.id,
      active: false
    }).then(res => {
      if (res?.data?.updateGitlabProvidedRepositoryActive) {
        setInactiveRepositories(sortRepos([...inactiveRepositories, repo]))
        const nextPage = isLastItem ? page - 1 : page
        loadPage(nextPage || 1)
      }
    })
  }

  const loadPage = async (pageNo: number) => {
    try {
      setFetching(true)
      const res = await fetchRepositoriesSequentially(pageNo)
      setActiveRepositoriesResult(res)
      setPage(pageNo)
    } catch (e) {
    } finally {
      setFetching(false)
    }
  }

  const clearRecentlyActivated = useDebounceCallback((page: number) => {
    setRecentlyActivatedRepositories([])
    loadPage(page)
  }, 3000)

  const [open, setOpen] = React.useState(false)

  const sortRepos = (repos: GitlabRepositories) => {
    if (!repos?.length) return repos
    return repos.sort((a, b) => a.node.name?.localeCompare(b.node.name))
  }

  const onCreated = (id: string) => {
    const activedRepo = inactiveRepositories?.find(o => o?.node?.id === id)
    if (activedRepo) {
      setRecentlyActivatedRepositories([
        activedRepo,
        ...recentlyActivatedRepositories
      ])
      setInactiveRepositories(repos =>
        sortRepos(repos.filter(o => o.node.id !== id))
      )
      clearRecentlyActivated.run(page)
    }
    setOpen(false)
  }

  const handleLoadPage = (page: number) => {
    clearRecentlyActivated.cancel()
    setRecentlyActivatedRepositories([])
    loadPage(page)
  }

  React.useEffect(() => {
    loadPage(1)

    return () => {
      clearRecentlyActivated.cancel()
    }
  }, [])

  return (
    <LoadingWrapper loading={fetching}>
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
                  <AddRepositoryForm
                    onCancel={() => setOpen(false)}
                    onCreated={onCreated}
                    repositories={inactiveRepositories}
                    kind={RepositoryKind.Gitlab}
                    providerStatus={providerStatus}
                    fetchingRepos={!isInactiveRepositoriesLoaded}
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
          {activeRepos?.length || recentlyActivatedRepositories?.length ? (
            <>
              {recentlyActivatedRepositories?.map(x => {
                return (
                  <TableRow key={x.node.id} className="!bg-muted/80">
                    <TableCell>{x.node.name}</TableCell>
                    <TableCell>{x.node.gitUrl}</TableCell>
                    <TableCell className="flex justify-end">
                      <div
                        className={buttonVariants({
                          variant: 'ghost',
                          size: 'icon'
                        })}
                      >
                        <IconSpinner />
                      </div>
                    </TableCell>
                  </TableRow>
                )
              })}
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
                          onClick={e => handleDelete(x)}
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
      {(page > 1 || pageInfo?.hasNextPage) && (
        <div className="mt-2 flex justify-end">
          <div className="flex w-[100px] items-center justify-center text-sm font-medium">
            {' '}
            Page {page}
          </div>
          <div className="flex items-center space-x-2">
            <Button
              variant="outline"
              className="h-8 w-8 p-0"
              disabled={fetching || page === 1}
              onClick={e => {
                handleLoadPage(page - 1)
              }}
            >
              <IconChevronLeft className="h-4 w-4" />
            </Button>
            <Button
              variant="outline"
              className="h-8 w-8 p-0"
              disabled={fetching || !pageInfo?.hasNextPage}
              onClick={e => {
                handleLoadPage(page + 1)
              }}
            >
              <IconChevronRight className="h-4 w-4" />
            </Button>
          </div>
        </div>
      )}
    </LoadingWrapper>
  )
}

function useAllInactiveRepositories(id: string) {
  const [queryVariables, setQueryVariables] = useState<
    QueryVariables<typeof listGitlabRepositories>
  >({ providerIds: [id], first: DEFAULT_PAGE_SIZE, active: false })
  const [repositories, setRepositories] = useState<GitlabRepositories>([])
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
          after: pageInfo.endCursor,
          active: false
        })
      } else {
        setIsAllLoaded(true)
      }
    }
  }, [fetching, data])

  return {
    repositories,
    setRepositories,
    isAllLoaded
  }
}

export default GitlabProviderDetail
