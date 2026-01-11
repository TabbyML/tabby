'use client'

import React from 'react'
import { zodResolver } from '@hookform/resolvers/zod'
import { useForm } from 'react-hook-form'
import { toast } from 'sonner'
import useSWR from 'swr'
import { useQuery } from 'urql'
import * as z from 'zod'

import { DEFAULT_PAGE_SIZE } from '@/lib/constants'
import { graphql } from '@/lib/gql/generates'
import { useMutation } from '@/lib/tabby/gql'
import { listRepositories, userGroupsQuery } from '@/lib/tabby/query'
import { Button } from '@/components/ui/button'
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
import { IconPencil, IconTrash } from '@/components/ui/icons'
import { Input } from '@/components/ui/input'
import {
  Pagination,
  PaginationContent,
  PaginationItem,
  PaginationNext,
  PaginationPrevious
} from '@/components/ui/pagination'
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
import LoadingWrapper from '@/components/loading-wrapper'

import { AccessPolicyView } from '../../components/access-policy-view'
import { JobInfoView } from '../../components/job-trigger'
import { triggerJobRunMutation } from '../../query'

const deleteRepositoryMutation = graphql(/* GraphQL */ `
  mutation deleteGitRepository($id: ID!) {
    deleteGitRepository(id: $id)
  }
`)

const updateRepositoryMutation = graphql(/* GraphQL */ `
  mutation updateGitRepository($id: ID!, $refs: [String!]) {
    updateGitRepository(id: $id, refs: $refs)
  }
`)

const formSchema = z.object({
  name: z.string(),
  gitUrl: z.string(),
  refs: z.array(z.string()).optional()
})

type FormValues = z.infer<typeof formSchema>

const PAGE_SIZE = DEFAULT_PAGE_SIZE

export default function RepositoryTable() {
  const [before, setBefore] = React.useState<string | undefined>()
  const [{ data, fetching }, reexecuteQuery] = useQuery({
    query: listRepositories,
    variables: { last: PAGE_SIZE, before }
  })

  useSWR(
    ['refresh_repos', before],
    () => {
      reexecuteQuery()
    },
    {
      revalidateOnFocus: true,
      revalidateOnReconnect: true,
      revalidateOnMount: false,
      refreshInterval: 10 * 1000
    }
  )

  const [{ data: userGroupData, fetching: fetchingUserGroups }] = useQuery({
    query: userGroupsQuery
  })

  const [currentPage, setCurrentPage] = React.useState(1)
  const edges = React.useMemo(() => {
    return data?.gitRepositories?.edges?.slice().reverse()
  }, [data?.gitRepositories?.edges])
  const pageInfo = data?.gitRepositories?.pageInfo
  const pageNum = Math.ceil((edges?.length || 0) / PAGE_SIZE)

  const getBeforeCursor = (page: number) => {
    return edges?.slice(0, (page - 1) * PAGE_SIZE)?.pop()?.cursor
  }

  const fetchPage = (page: number) => {
    setBefore(getBeforeCursor(page))
  }

  const currentPageRepos = React.useMemo(() => {
    return edges?.slice?.(
      (currentPage - 1) * PAGE_SIZE,
      currentPage * PAGE_SIZE
    )
  }, [currentPage, edges])

  const hasNextPage = pageInfo?.hasPreviousPage || currentPage < pageNum
  const hasPrevPage = currentPage > 1
  const showPagination =
    !!currentPageRepos?.length && (hasNextPage || hasPrevPage)

  const deleteRepository = useMutation(deleteRepositoryMutation)
  const updateRepository = useMutation(updateRepositoryMutation)
  const triggerJobRun = useMutation(triggerJobRunMutation)
  const [editingRepo, setEditingRepo] = React.useState<{
    id: string
    name: string
    gitUrl: string
    refs: string[]
  } | null>(null)

  const handleNavToPrevPage = () => {
    if (currentPage <= 1) return
    if (fetching) return

    const prevPage = currentPage - 1
    fetchPage(prevPage)
    setCurrentPage(prevPage)
  }

  const handleNavToNextPage = () => {
    if (!hasNextPage) return
    if (fetching) return

    const nextPage = currentPage + 1
    fetchPage(nextPage)
    setCurrentPage(nextPage)
  }

  const handleDeleteRepository = (id: string, isLast: boolean) => {
    deleteRepository({ id }).then(res => {
      if (res?.data?.deleteGitRepository) {
        fetchPage(isLast ? currentPage - 1 : currentPage)
      } else {
        toast.error(res?.error?.message || 'Failed to delete repository')
      }
    })
  }

  const handleTriggerJobRun = (command: string) => {
    return triggerJobRun({ command }).then(res => {
      if (res?.data?.triggerJobRun) {
        toast.success(
          'The job has been triggered successfully, it may take a few minutes to process.'
        )
        reexecuteQuery()
      } else {
        toast.error(res?.error?.message || 'Failed to trigger job')
      }
    })
  }

  const handleEditRepository = (repo: {
    id: string
    name: string
    gitUrl: string
    refs: Array<{ name: string }>
  }) => {
    setEditingRepo({
      id: repo.id,
      name: repo.name,
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

  const handleUpdateRepository = (values: FormValues) => {
    if (!editingRepo) return

    updateRepository({
      id: editingRepo.id,
      refs: values.refs && values.refs.length > 0 ? values.refs : undefined
    }).then(res => {
      if (res?.data?.updateGitRepository) {
        toast.success('Repository updated successfully')
        setEditingRepo(null)
        reexecuteQuery()
      } else {
        toast.error(res?.error?.message || 'Failed to update repository')
      }
    })
  }

  React.useEffect(() => {
    if (fetching) return
    if (pageNum < currentPage && currentPage > 1) {
      setCurrentPage(pageNum)
    }
  }, [pageNum, currentPage])

  return (
    <>
      <LoadingWrapper loading={fetching}>
        <ScrollArea>
          <Table className="min-w-[400px] border-t">
            <TableHeader>
              <TableRow>
                <TableHead className="w-[25%]">Name</TableHead>
                <TableHead className="w-[45%]">Git URL</TableHead>
                <TableHead className="w-[140px]">Access</TableHead>
                <TableHead>Job</TableHead>
                <TableHead className="w-[100px]"></TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {!currentPageRepos?.length && currentPage === 1 ? (
                <TableRow>
                  <TableCell colSpan={5} className="h-[100px] text-center">
                    No Data
                  </TableCell>
                </TableRow>
              ) : (
                <>
                  {currentPageRepos?.map(x => {
                    return (
                      <TableRow key={x.node.id}>
                        <TableCell
                          className="break-all lg:break-words"
                          title={x.node.name}
                        >
                          {x.node.name}
                        </TableCell>
                        <TableCell
                          className="break-all lg:break-words"
                          title={x.node.gitUrl}
                        >
                          {x.node.gitUrl}
                        </TableCell>
                        <TableCell className="break-all lg:break-words">
                          <AccessPolicyView
                            sourceId={x.node.sourceId}
                            sourceName={x.node.name}
                            fetchingUserGroups={fetchingUserGroups}
                            userGroups={userGroupData?.userGroups}
                            editable
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
                                  name: x.node.name,
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
                              onClick={() =>
                                handleDeleteRepository(
                                  x.node.id,
                                  currentPageRepos.length === 1
                                )
                              }
                            >
                              <IconTrash />
                            </Button>
                          </div>
                        </TableCell>{' '}
                      </TableRow>
                    )
                  })}
                </>
              )}
            </TableBody>
          </Table>
          <ScrollBar orientation="horizontal" />
        </ScrollArea>
        {showPagination && (
          <Pagination className="my-4">
            <PaginationContent>
              <PaginationItem>
                <PaginationPrevious
                  disabled={!hasPrevPage}
                  onClick={handleNavToPrevPage}
                />
              </PaginationItem>
              <PaginationItem>
                <PaginationNext
                  disabled={!hasNextPage}
                  onClick={handleNavToNextPage}
                />
              </PaginationItem>
            </PaginationContent>
          </Pagination>
        )}
      </LoadingWrapper>
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

function EditRepositoryDialog({
  repo,
  open,
  onOpenChange,
  onSubmit
}: {
  repo: { id: string; name: string; gitUrl: string; refs: string[] } | null
  open: boolean
  onOpenChange: (open: boolean) => void
  onSubmit: (values: FormValues) => void
}) {
  const form = useForm<FormValues>({
    resolver: zodResolver(formSchema)
  })

  React.useEffect(() => {
    if (repo) {
      form.reset({
        name: repo.name,
        gitUrl: repo.gitUrl,
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
            <FormField
              control={form.control}
              name="name"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Name</FormLabel>
                  <FormControl>
                    <Input
                      placeholder="e.g. tabby"
                      autoCapitalize="none"
                      autoCorrect="off"
                      {...field}
                      disabled={true}
                    />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="gitUrl"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Git URL</FormLabel>
                  <FormDescription>Remote or local Git URL</FormDescription>
                  <FormControl>
                    <Input
                      placeholder="e.g. https://github.com/TabbyML/tabby"
                      autoCapitalize="none"
                      autoCorrect="off"
                      {...field}
                      disabled={true}
                    />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
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
