'use client'

import React from 'react'
import { toast } from 'sonner'
import { useQuery } from 'urql'

import { DEFAULT_PAGE_SIZE } from '@/lib/constants'
import { graphql } from '@/lib/gql/generates'
import { useMutation } from '@/lib/tabby/gql'
import { listWebCrawlerUrl } from '@/lib/tabby/query'
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

import { JobInfoView } from '../../components/job-trigger'
import { triggerJobRunMutation } from '../../query'

const deleteWebCrawlerUrlMutation = graphql(/* GraphQL */ `
  mutation DeleteWebCrawlerUrl($id: ID!) {
    deleteWebCrawlerUrl(id: $id)
  }
`)

const PAGE_SIZE = DEFAULT_PAGE_SIZE

export default function WebCrawlerTable() {
  const [before, setBefore] = React.useState<string | undefined>()
  const [{ data, fetching }, reexecuteQuery] = useQuery({
    query: listWebCrawlerUrl,
    variables: { last: PAGE_SIZE, before }
  })

  const [currentPage, setCurrentPage] = React.useState(1)
  const edges = React.useMemo(() => {
    return data?.webCrawlerUrls?.edges?.slice().reverse()
  }, [data?.webCrawlerUrls?.edges])
  const pageInfo = data?.webCrawlerUrls?.pageInfo
  const pageNum = Math.ceil((edges?.length || 0) / PAGE_SIZE)

  const currentPageUrls = React.useMemo(() => {
    return edges?.slice?.(
      (currentPage - 1) * PAGE_SIZE,
      currentPage * PAGE_SIZE
    )
  }, [currentPage, edges])

  const hasNextPage = pageInfo?.hasPreviousPage || currentPage < pageNum
  const hasPrevPage = currentPage > 1
  const showPagination =
    !!currentPageUrls?.length && (hasNextPage || hasPrevPage)

  const getBeforeCursor = (page: number) => {
    return edges?.slice(0, (page - 1) * PAGE_SIZE)?.pop()?.cursor
  }

  const fetchPage = (page: number) => {
    setBefore(getBeforeCursor(page))
  }

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

  const triggerJobRun = useMutation(triggerJobRunMutation)
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

  const deleteWebCrawlerUrl = useMutation(deleteWebCrawlerUrlMutation)
  const handleDeleteWebCrawler = (id: string, isLast: boolean) => {
    deleteWebCrawlerUrl({ id }).then(res => {
      if (res?.data?.deleteWebCrawlerUrl) {
        fetchPage(isLast ? currentPage - 1 : currentPage)
      } else {
        toast.error(res?.error?.message || 'Failed to delete')
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
    <LoadingWrapper loading={fetching}>
      <Table className="table-fixed border-b">
        <TableHeader>
          <TableRow>
            <TableHead className="w-[70%]">URL</TableHead>
            <TableHead>Job</TableHead>
            <TableHead className="w-[100px]"></TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {!currentPageUrls?.length && currentPage === 1 ? (
            <TableRow>
              <TableCell colSpan={4} className="h-[100px] text-center">
                No Data
              </TableCell>
            </TableRow>
          ) : (
            <>
              {currentPageUrls?.map(x => {
                return (
                  <TableRow key={x.node.id}>
                    <TableCell className="break-all lg:break-words">
                      {x.node.url}
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
                          handleDeleteWebCrawler(
                            x.node.id,
                            currentPageUrls.length === 1
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
  )
}
