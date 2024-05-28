'use client'

import React, { useEffect, useState } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { toast } from 'sonner'
import { TypedDocumentNode, useQuery } from 'urql'

import { DEFAULT_PAGE_SIZE } from '@/lib/constants'
import {
  ListGithubRepositoriesQuery,
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
  listGithubRepositories,
  listGithubRepositoryProviders,
  listGithubSelfHostedRepositories,
  listGithubSelfHostedRepositoryProviders,
  listGitlabRepositories,
  listGitlabRepositoryProviders,
  listGitlabSelfHostedRepositories,
  listGitlabSelfHostedRepositoryProviders
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

import {
  updateGithubProvidedRepositoryActiveMutation,
  updateGithubSelfHostedProvidedRepositoryActiveMutation,
  updateGitlabProvidedRepositoryActiveMutation,
  updateGitlabSelfHostedProvidedRepositoryActiveMutation
} from '../query'
import AddRepositoryForm from './add-repository-form'
import { UpdateProviderForm } from './update-provider-form'
import { useRepositoryKind } from '../../hooks/use-repository-kind'

const PAGE_SIZE = DEFAULT_PAGE_SIZE

type Repositories = Array<{
  cursor: string
  node: {
    id: string
    vendorId: string
    name: string
    gitUrl: string
    active: boolean
  }
}>

type ListRepositoriesResponseData = {
  edges: Repositories
  pageInfo: ListGithubRepositoriesQuery['githubRepositories']['pageInfo']
}

const ProviderDetail: React.FC = () => {
  const searchParams = useSearchParams()
  const kind = useRepositoryKind()
  const router = useRouter()
  const id = searchParams.get('id')?.toString() ?? ''

  const { query, resolver } = React.useMemo(() => {
    switch (kind) {
      case RepositoryKind.Github:
        return {
          query: listGithubRepositoryProviders,
          resolver: (
            res: QueryResponseData<typeof listGithubRepositoryProviders>
          ) => res?.githubRepositoryProviders?.edges
        }
      case RepositoryKind.GithubSelfHosted:
        return {
          query: listGithubSelfHostedRepositoryProviders,
          resolver: (
            res?: QueryResponseData<
              typeof listGithubSelfHostedRepositoryProviders
            >
          ) => res?.githubSelfHostedRepositoryProviders?.edges
        }
      case RepositoryKind.Gitlab:
        return {
          query: listGitlabRepositoryProviders,
          resolver: (
            res: QueryResponseData<typeof listGitlabRepositoryProviders>
          ) => res?.gitlabRepositoryProviders?.edges
        }
      case RepositoryKind.GitlabSelfHosted:
        return {
          query: listGitlabSelfHostedRepositoryProviders,
          resolver: (
            res?: QueryResponseData<
              typeof listGitlabSelfHostedRepositoryProviders
            >
          ) => res?.gitlabSelfHostedRepositoryProviders?.edges
        }
      default:
        return {
          query: listGithubRepositoryProviders,
          resolver: (
            res?: QueryResponseData<typeof listGithubRepositoryProviders>
          ) => res?.githubRepositoryProviders?.edges
        }
    }
  }, [kind])

  const [{ data, fetching }, reexecuteQuery] = useQuery({
    query: query as TypedDocumentNode<any, any>,
    variables: { ids: [id] },
    pause: !id
  })
  const provider = resolver(data)?.[0]?.node

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
            kind={kind}
          />
        </LoadingWrapper>
      </CardContent>

      <div className="p-4">
        <ActiveRepoTable
          kind={kind}
          providerStatus={provider?.status}
          providerId={id}
        />
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
  kind: RepositoryKind
}> = ({ providerStatus, providerId, kind }) => {
  const { query, resolver, updateQuery, updateResolver } = React.useMemo(() => {
    switch (kind) {
      case RepositoryKind.Github:
        return {
          query: listGithubRepositories,
          updateQuery: updateGithubProvidedRepositoryActiveMutation,
          resolver: (res?: QueryResponseData<typeof listGithubRepositories>) =>
            res?.githubRepositories,
          updateResolver: (
            res?: QueryResponseData<
              typeof updateGithubProvidedRepositoryActiveMutation
            >
          ) => res?.updateGithubProvidedRepositoryActive
        }
      case RepositoryKind.GithubSelfHosted:
        return {
          query: listGithubSelfHostedRepositories,
          updateQuery: updateGithubSelfHostedProvidedRepositoryActiveMutation,
          resolver: (
            res?: QueryResponseData<typeof listGithubSelfHostedRepositories>
          ) => res?.githubSelfHostedRepositories,
          updateResolver: (
            res?: QueryResponseData<
              typeof updateGithubSelfHostedProvidedRepositoryActiveMutation
            >
          ) => res?.updateGithubSelfHostedProvidedRepositoryActive
        }
      case RepositoryKind.Gitlab:
        return {
          query: listGitlabRepositories,
          updateQuery: updateGitlabProvidedRepositoryActiveMutation,
          resolver: (res?: QueryResponseData<typeof listGitlabRepositories>) =>
            res?.gitlabRepositories,
          updateResolver: (
            res?: QueryResponseData<
              typeof updateGithubSelfHostedProvidedRepositoryActiveMutation
            >
          ) => res?.updateGithubSelfHostedProvidedRepositoryActive
        }
      case RepositoryKind.GitlabSelfHosted:
        return {
          query: listGitlabSelfHostedRepositories,
          updateQuery: updateGitlabSelfHostedProvidedRepositoryActiveMutation,
          resolver: (
            res?: QueryResponseData<typeof listGitlabSelfHostedRepositories>
          ) => res?.gitlabSelfHostedRepositories,
          updateResolver: (
            res?: QueryResponseData<
              typeof updateGitlabSelfHostedProvidedRepositoryActiveMutation
            >
          ) => res?.updateGitlabSelfHostedProvidedRepositoryActive
        }
      default:
        return {
          query: listGithubRepositories,
          updateQuery: updateGithubProvidedRepositoryActiveMutation,
          resolver: (res?: QueryResponseData<typeof listGithubRepositories>) =>
            res?.githubRepositories,
          updateResolver: (
            res?: QueryResponseData<
              typeof updateGithubProvidedRepositoryActiveMutation
            >
          ) => res?.updateGithubProvidedRepositoryActive
        }
    }
  }, [kind]) as {
    query: TypedDocumentNode<ListRepositoriesResponseData>
    updateQuery: TypedDocumentNode
    resolver: (res?: QueryResponseData<any>) => ListRepositoriesResponseData
    updateResolver: (res?: QueryResponseData<any>) => boolean | undefined
  }

  const [page, setPage] = React.useState(1)
  const {
    repositories: inactiveRepositories,
    setRepositories: setInactiveRepositories,
    isAllLoaded: isInactiveRepositoriesLoaded
  } = useAllInactiveRepositories(providerId, query, resolver)

  const fetchRepositories = (variables: QueryVariables<typeof query>) => {
    return client.query(query, variables).toPromise()
  }

  const fetchRepositoriesSequentially = async (
    page: number,
    cursor?: string
  ): Promise<ListRepositoriesResponseData | undefined> => {
    const res = await fetchRepositories({
      providerIds: [providerId],
      first: PAGE_SIZE,
      after: cursor,
      active: true
    })
    const responseData = resolver(res?.data)
    const _pageInfo = responseData?.pageInfo
    if (page - 1 > 0 && _pageInfo?.hasNextPage && _pageInfo?.endCursor) {
      return fetchRepositoriesSequentially(page - 1, _pageInfo.endCursor)
    } else {
      return res?.data
    }
  }

  const [activeRepositoriesResult, setActiveRepositoriesResult] =
    React.useState<QueryResponseData<typeof query>>()
  const [fetching, setFetching] = React.useState(true)
  const [recentlyActivatedRepositories, setRecentlyActivatedRepositories] =
    React.useState<Repositories>([])
  const activeRepos = resolver?.(activeRepositoriesResult)?.edges
  const pageInfo = resolver?.(activeRepositoriesResult)?.pageInfo

  const updateProvidedRepositoryActive = useMutation(updateQuery, {
    onError(error) {
      toast.error(error.message || 'Failed to delete')
    }
  })

  const handleDelete = async (repo: Repositories[0], isLastItem?: boolean) => {
    updateProvidedRepositoryActive({
      id: repo.node.id,
      active: false
    }).then(res => {
      if (updateResolver?.(res?.data)) {
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

  const sortRepos = (repos: Repositories) => {
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
                <DialogContent className="top-[20vh]">
                  <DialogHeader className="gap-3">
                    <DialogTitle>Add new repository</DialogTitle>
                    <DialogDescription>
                      Add new GitHub repository from this provider
                    </DialogDescription>
                  </DialogHeader>
                  <AddRepositoryForm
                    onCancel={() => setOpen(false)}
                    onCreated={onCreated}
                    repositories={inactiveRepositories}
                    kind={RepositoryKind.Github}
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
                          onClick={e =>
                            handleDelete(x, activeRepos?.length === 1)
                          }
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

function useAllInactiveRepositories(
  id: string,
  query: TypedDocumentNode<any, any>,
  resolver: (
    res?: QueryResponseData<ListRepositoriesResponseData>
  ) => ListRepositoriesResponseData
) {
  const [queryVariables, setQueryVariables] = useState<
    QueryVariables<typeof query>
  >({ providerIds: [id], first: PAGE_SIZE, active: false })
  const [repositories, setRepositories] = useState<Repositories>([])
  const [isAllLoaded, setIsAllLoaded] = useState(!id)

  const [{ data, fetching }] = useQuery({
    query,
    variables: queryVariables,
    pause: !id
  })

  useEffect(() => {
    if (isAllLoaded) return
    if (!fetching && data) {
      const pageInfo = resolver(data)?.pageInfo
      const currentList = [...repositories]
      setRepositories(currentList.concat(resolver(data)?.edges))

      if (pageInfo?.hasNextPage) {
        setQueryVariables({
          providerIds: [id],
          first: PAGE_SIZE,
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

export default ProviderDetail
