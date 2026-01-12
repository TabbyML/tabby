'use client'

import React, { useEffect, useState } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { zodResolver } from '@hookform/resolvers/zod'
import { useForm } from 'react-hook-form'
import { toast } from 'sonner'
import useSWR from 'swr'
import { useQuery } from 'urql'
import * as z from 'zod'

import { DEFAULT_PAGE_SIZE } from '@/lib/constants'
import {
  IntegrationKind,
  IntegrationStatus,
  ListIntegratedRepositoriesQuery,
  ListIntegrationsQuery
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
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage
} from '@/components/ui/form'
import {
  IconChevronLeft,
  IconChevronRight,
  IconPencil,
  IconPlus,
  IconSpinner,
  IconTrash
} from '@/components/ui/icons'
import { Input } from '@/components/ui/input'
import { ScrollArea, ScrollBar } from '@/components/ui/scroll-area'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow
} from '@/components/ui/table'
import { TagInput } from '@/components/ui/tag-input'
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger
} from '@/components/ui/tooltip'
import LoadingWrapper from '@/components/loading-wrapper'
import { ListSkeleton } from '@/components/skeleton'

import { AccessPolicyView } from '../../../components/access-policy-view'
import { JobInfoView } from '../../../components/job-trigger'
import { triggerJobRunMutation } from '../../../query'
import { useIntegrationKind } from '../../hooks/use-repository-kind'
import {
  updateIntegratedRepositoryActiveMutation,
  updateIntegratedRepositoryRefsMutation
} from '../query'
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
  const shouldRefreshProvider = provider?.status === IntegrationStatus.Pending

  useSWR(
    shouldRefreshProvider ? 'refresh' : null,
    () => {
      reexecuteQuery()
    },
    {
      revalidateOnFocus: true,
      revalidateOnReconnect: true,
      revalidateOnMount: false,
      refreshInterval: 5 * 1000
    }
  )

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
          <div className="ml-1">{provider && toStatusBadge(provider)}</div>
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

  useSWR(
    ['refresh_repos', page],
    ([, p]) => {
      fetchRepositoriesSequentially(p).then(res =>
        setActiveRepositoriesResult(res)
      )
    },
    {
      revalidateOnFocus: true,
      revalidateOnReconnect: true,
      revalidateOnMount: false,
      refreshInterval: 10 * 1000
    }
  )

  const [activeRepositoriesResult, setActiveRepositoriesResult] =
    React.useState<QueryResponseData<typeof listIntegratedRepositories>>()
  const [fetching, setFetching] = React.useState(true)
  const [recentlyActivatedRepositories, setRecentlyActivatedRepositories] =
    React.useState<IntegratedRepositories>([])
  const activeRepos = activeRepositoriesResult?.integratedRepositories?.edges
  const pageInfo = activeRepositoriesResult?.integratedRepositories?.pageInfo
  const [editingRepo, setEditingRepo] = React.useState<{
    id: string
    displayName: string
    gitUrl: string
    refs: string[]
  } | null>(null)

  const updateProvidedRepositoryActive = useMutation(
    updateIntegratedRepositoryActiveMutation,
    {
      onError(error) {
        toast.error(error.message || 'Failed to delete')
      }
    }
  )

  const updateProvidedRepositoryRefs = useMutation(
    updateIntegratedRepositoryRefsMutation,
    {
      onError(error) {
        toast.error(error.message || 'Failed to update')
      }
    }
  )

  const handleUpdateRepository = (values: { refs?: string[] }) => {
    if (!editingRepo) return

    updateProvidedRepositoryRefs({
      id: editingRepo.id,
      refs: values.refs && values.refs.length > 0 ? values.refs : []
    }).then(res => {
      if (res?.data?.updateIntegratedRepositoryRefs) {
        toast.success('Repository updated successfully')
        setEditingRepo(null)
        loadPage(page)
      }
    })
  }

  const handleEditRepository = (repo: {
    id: string
    displayName: string
    gitUrl: string
    refs: Array<{ name: string }>
  }) => {
    setEditingRepo({
      id: repo.id,
      displayName: repo.displayName,
      gitUrl: repo.gitUrl,
      refs: repo.refs.map(r => {
        // Extract branch name from refs/heads/xxx or refs/tags/xxx
        if (r.name.startsWith('refs/heads/')) {
          return r.name.substring('refs/heads/'.length)
        } else if (r.name.startsWith('refs/tags/')) {
          return r.name.substring('refs/tags/'.length)
        }
        return r.name
      })
    })
  }

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
        <ScrollArea>
          <Table className="min-w-[400px]">
            <TableHeader>
              <TableRow>
                <TableHead className="w-[25%]">Name</TableHead>
                <TableHead className="w-[35%]">URL</TableHead>
                <TableHead className="w-[140px]">Access</TableHead>
                <TableHead>Job</TableHead>
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
                          <div className="flex justify-end gap-2">
                            <Button
                              size="icon"
                              variant="ghost"
                              onClick={() =>
                                handleEditRepository({
                                  id: x.node.id,
                                  displayName: x.node.displayName,
                                  gitUrl: x.node.gitUrl,
                                  refs: x.node.refs
                                })
                              }
                            >
                              <IconPencil />
                            </Button>
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
          <ScrollBar orientation="horizontal" />
        </ScrollArea>
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
        <DialogContent>
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
      <EditRepositoryDialog
        repo={editingRepo}
        open={!!editingRepo}
        onOpenChange={open => {
          if (!open) setEditingRepo(null)
        }}
        onSubmit={handleUpdateRepository}
      />
    </>
  )
}

const editFormSchema = z.object({
  refs: z.array(z.string()).optional()
})

type EditFormValues = z.infer<typeof editFormSchema>

function EditRepositoryDialog({
  repo,
  open,
  onOpenChange,
  onSubmit
}: {
  repo: {
    id: string
    displayName: string
    gitUrl: string
    refs: string[]
  } | null
  open: boolean
  onOpenChange: (open: boolean) => void
  onSubmit: (values: EditFormValues) => void
}) {
  const form = useForm<EditFormValues>({
    resolver: zodResolver(editFormSchema)
  })

  React.useEffect(() => {
    if (repo) {
      form.reset({
        refs: repo.refs
      })
    }
  }, [repo, form])

  const { isSubmitting } = form.formState

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-xl">
        <DialogHeader>
          <DialogTitle>Edit Repository</DialogTitle>
          <DialogDescription>
            Update the repository branches to index
          </DialogDescription>
        </DialogHeader>
        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
            <FormItem>
              <FormLabel>Name</FormLabel>
              <FormControl>
                <Input
                  value={repo?.displayName}
                  disabled={true}
                  autoCapitalize="none"
                  autoCorrect="off"
                />
              </FormControl>
              <FormMessage />
            </FormItem>
            <FormItem>
              <FormLabel>Git URL</FormLabel>
              <FormDescription>Remote or local Git URL</FormDescription>
              <FormControl>
                <Input
                  value={repo?.gitUrl}
                  disabled={true}
                  autoCapitalize="none"
                  autoCorrect="off"
                />
              </FormControl>
              <FormMessage />
            </FormItem>
            <FormField
              control={form.control}
              name="refs"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Branches</FormLabel>
                  <FormDescription>
                    Branches to index (press Enter to select, leave empty for
                    default branch)
                  </FormDescription>
                  <FormControl>
                    <TagInput
                      placeholder="e.g. main"
                      autoCapitalize="none"
                      autoCorrect="off"
                      {...field}
                    />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            <div className="flex justify-end gap-4">
              <Button
                type="button"
                variant="ghost"
                disabled={isSubmitting}
                onClick={() => onOpenChange(false)}
              >
                Cancel
              </Button>
              <Button type="submit" disabled={isSubmitting}>
                Update
              </Button>
            </div>
          </form>
        </Form>
      </DialogContent>
    </Dialog>
  )
}

function toStatusBadge(
  node: ListIntegrationsQuery['integrations']['edges'][0]['node']
) {
  switch (node.status) {
    case IntegrationStatus.Ready:
      return <Badge variant="successful">Ready</Badge>
    case IntegrationStatus.Failed: {
      return (
        <Tooltip delayDuration={0}>
          <TooltipTrigger>
            <Badge variant="destructive">Error</Badge>
          </TooltipTrigger>
          <TooltipContent>
            {node.message ? (
              <div>
                <p className="mb-2">{node.message}</p>
                Please verify your context provider settings to resolve the
                issue
              </div>
            ) : (
              <p>
                Processing error. Please check if the access token is still
                valid
              </p>
            )}
          </TooltipContent>
        </Tooltip>
      )
    }
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
