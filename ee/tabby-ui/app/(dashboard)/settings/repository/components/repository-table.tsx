'use client'

import React from 'react'
import { toast } from 'sonner'
import { useClient, useQuery } from 'urql'

import { graphql } from '@/lib/gql/generates'
import {
  RepositoriesQueryVariables,
  RepositoryEdge
} from '@/lib/gql/generates/graphql'
import { useMutation } from '@/lib/tabby/gql'
import { listRepositories } from '@/lib/tabby/query'
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
import { CopyButton } from '@/components/copy-button'

import CreateRepositoryForm from './create-repository-form'
import { CardHeader, CardTitle } from '@/components/ui/card'

const deleteRepositoryMutation = graphql(/* GraphQL */ `
  mutation deleteRepository($id: ID!) {
    deleteRepository(id: $id)
  }
`)

const PAGE_SIZE = 20
export default function RepositoryTable() {
  const client = useClient()
  const [{ data, fetching }] = useQuery({
    query: listRepositories,
    variables: { first: PAGE_SIZE }
  })
  // if a new repo was created, fetching all records and navigating to the last page
  const [fetchingLastPage, setFetchingLastPage] = React.useState(false)

  const [currentPage, setCurrentPage] = React.useState(1)
  const edges = data?.repositories?.edges
  const pageInfo = data?.repositories?.pageInfo
  const pageNum = Math.ceil((edges?.length || 0) / PAGE_SIZE)

  const currentPageRepos = React.useMemo(() => {
    return edges?.slice?.(
      (currentPage - 1) * PAGE_SIZE,
      currentPage * PAGE_SIZE
    )
  }, [currentPage, edges])

  const hasNextPage = pageInfo?.hasNextPage || currentPage < pageNum
  const hasPrevPage = currentPage > 1

  const fetchRepositories = (variables: RepositoriesQueryVariables) => {
    return client.query(listRepositories, variables).toPromise()
  }

  const fetchRecordsSequentially = async (cursor?: string): Promise<number> => {
    const res = await fetchRepositories({ first: PAGE_SIZE, after: cursor })
    let count = res?.data?.repositories?.edges?.length || 0
    const _pageInfo = res?.data?.repositories?.pageInfo
    if (_pageInfo?.hasNextPage && _pageInfo?.endCursor) {
      // cacheExchange will merge the edges
      count = await fetchRecordsSequentially(_pageInfo.endCursor)
    }
    return count
  }

  const fetchAllRecords = async () => {
    try {
      setFetchingLastPage(true)
      const count = fetchRecordsSequentially(pageInfo?.endCursor ?? undefined)
      return count
    } catch (e) {
      return 0
    } finally {
      setFetchingLastPage(false)
    }
  }

  const deleteRepository = useMutation(deleteRepositoryMutation)

  const handleRepositoryCreated = async () => {
    toast.success('Repository created')
    fetchAllRecords().then(count => {
      setCurrentPage(getPageNumber(count))
    })
  }

  const handleNavToPrevPage = () => {
    if (currentPage <= 1) return
    if (fetchingLastPage || fetching) return
    setCurrentPage(p => p - 1)
  }

  const handleFetchNextPage = () => {
    if (!hasNextPage) return
    if (fetchingLastPage || fetching) return

    fetchRepositories({ first: PAGE_SIZE, after: pageInfo?.endCursor }).then(
      data => {
        if (data?.data?.repositories?.edges?.length) {
          setCurrentPage(p => p + 1)
        }
      }
    )
  }

  const handleDeleteRepository = (node: RepositoryEdge['node']) => {
    deleteRepository({ id: node.id }).then(res => {
      if (res?.error) {
        toast.error(res.error.message)
        return
      }
      if (res?.data?.deleteRepository) {
        toast.success(`${node.name} deleted`)
      }
    })
  }

  React.useEffect(() => {
    if (pageNum < currentPage && currentPage > 1) {
      setCurrentPage(pageNum)
    }
  }, [pageNum, currentPage])

  return (
    <div>
      <CardHeader>
        <CardTitle>Repositories</CardTitle>
      </CardHeader>
      <div className='p-4'>
        <Table className="border-b">
          {!!currentPageRepos?.length && (
            <TableHeader>
              <TableRow>
                <TableHead className="w-[25%]">Name</TableHead>
                <TableHead className="w-[45%]">Git Url</TableHead>
                <TableHead></TableHead>
              </TableRow>
            </TableHeader>
          )}
          <TableBody>
            {currentPageRepos?.map(x => {
              return (
                <TableRow key={x.node.id}>
                  <TableCell>{x.node.name}</TableCell>
                  <TableCell>
                    <div className="flex items-center gap-1">
                      {x.node.gitUrl}
                      <CopyButton value={x.node.gitUrl} />
                    </div>
                  </TableCell>
                  <TableCell className="flex justify-end">
                    <div className="flex gap-1">
                      <Button
                        size="icon"
                        variant="hover-destructive"
                        onClick={() => handleDeleteRepository(x.node)}
                      >
                        <IconTrash />
                      </Button>
                    </div>
                  </TableCell>
                </TableRow>
              )
            })}
          </TableBody>
        </Table>
        {(hasNextPage || hasPrevPage) && (
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

        <CreateRepositoryForm onCreated={handleRepositoryCreated} />
      </div>
    </div>
  )
}

function getPageNumber(count?: number) {
  return Math.ceil((count || 0) / PAGE_SIZE)
}
