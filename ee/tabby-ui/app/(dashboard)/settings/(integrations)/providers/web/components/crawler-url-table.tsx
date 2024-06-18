'use client'

import React from 'react'
import { toast } from 'sonner'
import { useClient, useQuery } from 'urql'

import { DEFAULT_PAGE_SIZE } from '@/lib/constants'
import { graphql } from '@/lib/gql/generates'
import { WebCrawlerUrlsQueryVariables } from '@/lib/gql/generates/graphql'
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

const deleteWebCrawlerUrlMutation = graphql(/* GraphQL */ `
  mutation DeleteWebCrawlerUrl($id: ID!) {
    deleteWebCrawlerUrl(id: $id)
  }
`)

const PAGE_SIZE = DEFAULT_PAGE_SIZE

export default function WebCrawlerTable() {
  const client = useClient()
  const [{ data, fetching }] = useQuery({
    query: listWebCrawlerUrl,
    variables: { first: PAGE_SIZE }
  })

  const [currentPage, setCurrentPage] = React.useState(1)
  const edges = data?.webCrawlerUrls?.edges
  const pageInfo = data?.webCrawlerUrls?.pageInfo
  const pageNum = Math.ceil((edges?.length || 0) / PAGE_SIZE)

  const currentPageUrls = React.useMemo(() => {
    return edges?.slice?.(
      (currentPage - 1) * PAGE_SIZE,
      currentPage * PAGE_SIZE
    )
  }, [currentPage, edges])

  const hasNextPage = pageInfo?.hasNextPage || currentPage < pageNum
  const hasPrevPage = currentPage > 1
  const showPagination =
    !!currentPageUrls?.length && (hasNextPage || hasPrevPage)

  const fetchWebCrawlerUrls = (variables: WebCrawlerUrlsQueryVariables) => {
    return client.query(listWebCrawlerUrl, variables).toPromise()
  }

  const fetchRecordsSequentially = async (cursor?: string): Promise<number> => {
    const res = await fetchWebCrawlerUrls({ first: PAGE_SIZE, after: cursor })
    let count = res?.data?.webCrawlerUrls?.edges?.length || 0
    const _pageInfo = res?.data?.webCrawlerUrls?.pageInfo
    if (_pageInfo?.hasNextPage && _pageInfo?.endCursor) {
      // cacheExchange will merge the edges
      count = await fetchRecordsSequentially(_pageInfo.endCursor)
    }
    return count
  }

  const deleteWebCrawlerUrl = useMutation(deleteWebCrawlerUrlMutation)

  const handleNavToPrevPage = () => {
    if (currentPage <= 1) return
    if (fetching) return
    setCurrentPage(p => p - 1)
  }

  const handleFetchNextPage = () => {
    if (!hasNextPage) return
    if (fetching) return

    fetchWebCrawlerUrls({ first: PAGE_SIZE, after: pageInfo?.endCursor }).then(
      data => {
        if (data?.data?.webCrawlerUrls?.edges?.length) {
          setCurrentPage(p => p + 1)
        }
      }
    )
  }

  const handleDeleteWebCrawler = (id: string) => {
    deleteWebCrawlerUrl({ id }).then(res => {
      if (res?.error) {
        toast.error(res.error.message)
        return
      }
    })
  }

  React.useEffect(() => {
    if (pageNum < currentPage && currentPage > 1) {
      setCurrentPage(pageNum)
    }
  }, [pageNum, currentPage])

  return (
    <LoadingWrapper loading={fetching}>
      <Table className="table-fixed border-b">
        <TableHeader>
          <TableRow>
            <TableHead>URL</TableHead>
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
                    <TableCell className="truncate">{x.node.url}</TableCell>
                    <TableCell className="flex justify-end">
                      <div className="flex gap-1">
                        <Button
                          size="icon"
                          variant="hover-destructive"
                          onClick={() => handleDeleteWebCrawler(x.node.id)}
                        >
                          <IconTrash />
                        </Button>
                      </div>
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
                onClick={handleFetchNextPage}
              />
            </PaginationItem>
          </PaginationContent>
        </Pagination>
      )}
    </LoadingWrapper>
  )
}
