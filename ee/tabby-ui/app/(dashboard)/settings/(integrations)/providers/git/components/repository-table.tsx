'use client'

import React from 'react'
import { toast } from 'sonner'
import { useQuery } from 'urql'

import { DEFAULT_PAGE_SIZE } from '@/lib/constants'
import { graphql } from '@/lib/gql/generates'
import { useMutation } from '@/lib/tabby/gql'
import { listRepositories, userGroupsQuery } from '@/lib/tabby/query'
import { Button } from '@/components/ui/button'
import { IconTrash } from '@/components/ui/icons'
import {
  Pagination,
  PaginationContent,
  PaginationItem,
  PaginationNext,
  PaginationPrevious
} from '@/components/ui/pagination'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow
} from '@/components/ui/table'
import LoadingWrapper from '@/components/loading-wrapper'

import { AccessPolicyView } from '../../components/access-policy-view'
import { JobInfoView } from '../../components/job-trigger'
import { triggerJobRunMutation } from '../../query'

const deleteRepositoryMutation = graphql(/* GraphQL */ `
  mutation deleteGitRepository($id: ID!) {
    deleteGitRepository(id: $id)
  }
`)

const PAGE_SIZE = DEFAULT_PAGE_SIZE

export default function RepositoryTable() {
  const [before, setBefore] = React.useState<string | undefined>()
  const [{ data, fetching }, reexecuteQuery] = useQuery({
    query: listRepositories,
    variables: { last: PAGE_SIZE, before }
  })

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
  const triggerJobRun = useMutation(triggerJobRunMutation)

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

  React.useEffect(() => {
    if (fetching) return
    if (pageNum < currentPage && currentPage > 1) {
      setCurrentPage(pageNum)
    }
  }, [pageNum, currentPage])

  return (
    <>
      <LoadingWrapper loading={fetching}>
        <Table className="table-fixed border-t">
          <TableHeader>
            <TableRow>
              <TableHead className="w-[25%]">Name</TableHead>
              <TableHead className="w-[45%]">Git URL</TableHead>
              <TableHead className="w-[140px]">Access</TableHead>
              <TableHead className="w-[180px]">Job</TableHead>
              <TableHead className="w-[60px]"></TableHead>
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
                      </TableCell>
                    </TableRow>
                  )
                })}
              </>
            )}
          </TableBody>
        </Table>
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
    </>
  )
}
