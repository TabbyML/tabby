'use client'

import React, { useEffect, useState } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { toast } from 'sonner'
import { useQuery } from 'urql'

import { DEFAULT_PAGE_SIZE } from '@/lib/constants'
import {
  IntegrationKind,
  IntegrationStatus,
  ListIntegratedRepositoriesQuery
} from '@/lib/gql/generates/graphql'
import { useDebounceCallback } from '@/lib/hooks/use-debounce'
import {
  client,
  QueryResponseData,
  QueryVariables,
  useMutation
} from '@/lib/tabby/gql'
import {
  listIntegratedRepositories,
  listIntegrations,
  userGroupsQuery
} from '@/lib/tabby/query'
import { Badge } from '@/components/ui/badge'
import { Button, buttonVariants } from '@/components/ui/button'
import { CardContent, CardTitle } from '@/components/ui/card'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import {
  IconChevronLeft,
  IconChevronRight,
  IconPlus,
  IconSpinner,
  IconTrash
} from '@/components/ui/icons'
import { ScrollArea, ScrollBar } from '@/components/ui/scroll-area'
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

import { AccessPolicyView } from '../../../components/access-policy-view'
import { JobInfoView } from '../../../components/job-trigger'
import { triggerJobRunMutation } from '../../../query'
import { useIntegrationKind } from '../../hooks/use-repository-kind'
import { updateIntegratedRepositoryActiveMutation } from '../query'
import AddRepositoryForm from './add-repository-form'
import { UpdateProviderForm } from './update-provider-form'

const PAGE_SIZE = DEFAULT_PAGE_SIZE

type IntegratedRepositories =
  ListIntegratedRepositoriesQuery['integratedRepositories']['edges']

const ProviderDetail: React.FC = () => {
  const searchParams = useSearchParams()
  const kind = useIntegrationKind()
  const router = useRouter()
  const id = searchParams.get('id')?.toString() ?? ''

  const [{ data, fetching }, reexecuteQuery] = useQuery({
    query: listIntegrations,
    variables: { ids: [id], kind },
    pause: !id || !kind
  })
  const provider = data?.integrations?.edges?.[0]?.node

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

      <ScrollArea>
        <ActiveRepoTable
          kind={kind}
          providerStatus={provider?.status}
          providerId={id}
        />
        <ScrollBar orientation="horizontal" />
      </ScrollArea>
    </LoadingWrapper>
  )
}

const ActiveRepoTable: React.FC<{
  providerId: string
  providerStatus: IntegrationStatus | undefined
  kind: IntegrationKind
}> = ({ providerStatus, providerId, kind }) => {
  const [page, setPage] = React.useState(1)
  const [{ data: userGroupData, fetching: fetchingUserGroups }] = useQuery({
    query: userGroupsQuery
  })
  const {
    repositories: inactiveRepositories,
    setRepositories: setInactiveRepositories,
    isAllLoaded: isInactiveRepositoriesLoaded
  } = useAllInactiveRepositories(providerId, kind)

  const fetchRepositories = (
    variables: QueryVariables<typeof listIntegratedRepositories>
  ) => {
    return client.query(listIntegratedRepositories, variables).toPromise()
  }

  const fetchRepositoriesSequentially = async (
    page: number,
    cursor?: string
  ): Promise<
    QueryResponseData<typeof listIntegratedRepositories> | undefined
  > => {
    const res = await fetchRepositories({
      ids: [providerId],
      first: PAGE_SIZE,
      after: cursor,
      active: true,
      kind
    })
    const responseData = res?.data?.integratedRepositories
    const _pageInfo = responseData?.pageInfo
    if (page - 1 > 0 && _pageInfo?.hasNextPage && _pageInfo?.endCursor) {
      return fetchRepositoriesSequentially(page - 1, _pageInfo.endCursor)
    } else {
      return res?.data
    }
  }

  const [activeRepositoriesResult, setActiveRepositoriesResult] =
    React.useState<QueryResponseData<typeof listIntegratedRepositories>>()
  const [fetching, setFetching] = React.useState(true)
  const [recentlyActivatedRepositories, setRecentlyActivatedRepositories] =
    React.useState<IntegratedRepositories>([])
  const activeRepos = activeRepositoriesResult?.integratedRepositories?.edges
  const pageInfo = activeRepositoriesResult?.integratedRepositories?.pageInfo

  const updateProvidedRepositoryActive = useMutation(
    updateIntegratedRepositoryActiveMutation,
    {
      onError(error) {
        toast.error(error.message || 'Failed to delete')
      }
    }
  )

  const triggerJobRun = useMutation(triggerJobRunMutation)

  const handleDelete = async (
    repo: IntegratedRepositories[0],
    isLastItem?: boolean
  ) => {
    updateProvidedRepositoryActive({
      id: repo.node.id,
      active: false
    }).then(res => {
      if (res?.data?.updateIntegratedRepositoryActive) {
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

  const sortRepos = (repos: IntegratedRepositories) => {
    if (!repos?.length) return repos
    return repos.sort((a, b) =>
      a.node.displayName?.localeCompare(b.node.displayName)
    )
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

  const handleTriggerJobRun = (command: string) => {
    return triggerJobRun({ command }).then(res => {
      if (res?.data?.triggerJobRun) {
        toast.success(
          'The job has been triggered successfully, it may take a few minutes to process.'
        )
        handleLoadPage(page)
      } else {
        toast.error(res?.error?.message || 'Failed to trigger job')
      }
    })
  }

  React.useEffect(() => {
    loadPage(1)

    return () => clearRecentlyActivated.cancel()
  }, [])

  return (
    <>
      <LoadingWrapper loading={fetching}>
        <Table className="table-fixed">
          <TableHeader>
            <TableRow>
              <TableHead className="w-[25%]">Name</TableHead>
              <TableHead className="w-[35%]">URL</TableHead>
              <TableHead className="w-[140px]">Access</TableHead>
              <TableHead className="w-[180px]">Job</TableHead>
              <TableHead className="w-[60px] text-right">
                <Button
                  variant="outline"
                  size="icon"
                  className="shadow-none"
                  onClick={e => setOpen(true)}
                >
                  <IconPlus />
                </Button>
              </TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {activeRepos?.length || recentlyActivatedRepositories?.length ? (
              <>
                {recentlyActivatedRepositories?.map(x => {
                  return (
                    <TableRow key={x.node.id} className="!bg-muted/80">
                      <TableCell className="break-all lg:break-words">
                        {x.node.displayName}
                      </TableCell>
                      <TableCell className="break-all lg:break-words">
                        {x.node.gitUrl}
                      </TableCell>
                      <TableCell></TableCell>
                      <TableCell></TableCell>
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
                      <TableCell className="break-all lg:break-words">
                        {x.node.displayName}
                      </TableCell>
                      <TableCell className="break-all lg:break-words">
                        {x.node.gitUrl}
                      </TableCell>
                      <TableCell className="break-all lg:break-words">
                        <AccessPolicyView
                          sourceId={x.node.sourceId}
                          sourceName={x.node.displayName}
                          editable
                          fetchingUserGroups={fetchingUserGroups}
                          userGroups={userGroupData?.userGroups}
                        />
                      </TableCell>
                      <TableCell>
                        <JobInfoView
                          jobInfo={x.node.jobInfo}
                          onTrigger={() =>
                            handleTriggerJobRun(x.node.jobInfo.command)
                          }
                        />
                      </TableCell>
                      <TableCell className="text-right">
                        <Button
                          size="icon"
                          variant="hover-destructive"
                          onClick={e =>
                            handleDelete(x, activeRepos?.length === 1)
                          }
                        >
                          <IconTrash />
                        </Button>
                      </TableCell>
                    </TableRow>
                  )
                })}
              </>
            ) : (
              <TableRow>
                <TableCell
                  colSpan={5}
                  className="h-[100px] text-center hover:bg-background"
                >
                  <div className="mt-4 flex flex-col items-center gap-4">
                    <span>No repositories</span>
                    <Button onClick={e => setOpen(true)} className="gap-1">
                      <IconPlus />
                      Add
                    </Button>
                  </div>
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
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent className="top-[20vh]">
          <DialogHeader className="gap-3">
            <DialogTitle>Add new repository</DialogTitle>
            <DialogDescription>
              Add new repository from this provider
            </DialogDescription>
          </DialogHeader>
          <AddRepositoryForm
            onCancel={() => setOpen(false)}
            onCreated={onCreated}
            repositories={inactiveRepositories}
            kind={kind}
            providerStatus={providerStatus}
            fetchingRepos={!isInactiveRepositoriesLoaded}
          />
        </DialogContent>
      </Dialog>
    </>
  )
}

function toStatusBadge(status: IntegrationStatus) {
  switch (status) {
    case IntegrationStatus.Ready:
      return <Badge variant="successful">Ready</Badge>
    case IntegrationStatus.Failed:
      return <Badge variant="destructive">Error</Badge>
    case IntegrationStatus.Pending:
      return <Badge>Pending</Badge>
  }
}

function useAllInactiveRepositories(id: string, kind: IntegrationKind) {
  const [queryVariables, setQueryVariables] = useState<
    QueryVariables<typeof listIntegratedRepositories>
  >({ ids: [id], first: PAGE_SIZE, active: false, kind })
  const [repositories, setRepositories] = useState<IntegratedRepositories>([])
  const [isAllLoaded, setIsAllLoaded] = useState(!id)

  const [{ data, fetching }] = useQuery({
    query: listIntegratedRepositories,
    variables: queryVariables,
    pause: !id
  })

  useEffect(() => {
    if (isAllLoaded) return
    if (!fetching && data) {
      const pageInfo = data?.integratedRepositories?.pageInfo
      const currentList = [...repositories]
      setRepositories(currentList.concat(data?.integratedRepositories?.edges))

      if (pageInfo?.hasNextPage) {
        setQueryVariables({
          ids: [id],
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
