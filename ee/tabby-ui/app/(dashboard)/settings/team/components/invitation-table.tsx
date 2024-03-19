'use client'

import React from 'react'
import moment from 'moment'
import { toast } from 'sonner'
import { useClient, useQuery } from 'urql'

import { DEFAULT_PAGE_SIZE } from '@/lib/constants'
import { graphql } from '@/lib/gql/generates'
import {
  InvitationEdge,
  ListInvitationsQueryVariables
} from '@/lib/gql/generates/graphql'
import { useExternalURL } from '@/lib/hooks/use-network-setting'
import { useMutation } from '@/lib/tabby/gql'
import { listInvitations } from '@/lib/tabby/query'
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
import LoadingWrapper from '@/components/loading-wrapper'

import CreateInvitationForm from './create-invitation-form'

const deleteInvitationMutation = graphql(/* GraphQL */ `
  mutation DeleteInvitation($id: ID!) {
    deleteInvitation(id: $id)
  }
`)

const PAGE_SIZE = DEFAULT_PAGE_SIZE
export default function InvitationTable() {
  const client = useClient()
  const [{ data, fetching }] = useQuery({
    query: listInvitations,
    variables: { first: PAGE_SIZE }
  })
  // if a new invitation was created, fetching all records and navigating to the last page
  const [fetchingLastPage, setFetchingLastPage] = React.useState(false)

  const [currentPage, setCurrentPage] = React.useState(1)
  const edges = data?.invitations?.edges
  const pageInfo = data?.invitations?.pageInfo
  const pageNum = Math.ceil((edges?.length || 0) / PAGE_SIZE)

  const currentPageInvits = React.useMemo(() => {
    return edges?.slice?.(
      (currentPage - 1) * PAGE_SIZE,
      currentPage * PAGE_SIZE
    )
  }, [currentPage, edges])

  const hasNextPage = pageInfo?.hasNextPage || currentPage < pageNum
  const hasPrevPage = currentPage > 1

  const fetchInvitations = (variables: ListInvitationsQueryVariables) => {
    return client.query(listInvitations, variables).toPromise()
  }

  const fetchInvitationsSequentially = async (
    cursor?: string
  ): Promise<number> => {
    const res = await fetchInvitations({ first: PAGE_SIZE, after: cursor })
    let count = res?.data?.invitations?.edges?.length || 0
    const _pageInfo = res?.data?.invitations?.pageInfo
    if (_pageInfo?.hasNextPage && _pageInfo?.endCursor) {
      // cacheExchange will merge the edges
      count = await fetchInvitationsSequentially(_pageInfo.endCursor)
    }
    return count
  }

  const fetchAllRecords = async () => {
    try {
      setFetchingLastPage(true)
      const count = fetchInvitationsSequentially(
        pageInfo?.endCursor ?? undefined
      )
      return count
    } catch (e) {
      return 0
    } finally {
      setFetchingLastPage(false)
    }
  }

  const externalUrl = useExternalURL()

  const deleteInvitation = useMutation(deleteInvitationMutation)

  const handleInvitationCreated = async () => {
    toast.success('Invitation created')
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

    fetchInvitations({ first: PAGE_SIZE, after: pageInfo?.endCursor }).then(
      data => {
        if (data?.data?.invitations?.edges?.length) {
          setCurrentPage(p => p + 1)
        }
      }
    )
  }

  const handleDeleteInvatation = (node: InvitationEdge['node']) => {
    deleteInvitation({ id: node.id }).then(res => {
      if (res?.error) {
        toast.error(res.error.message)
        return
      }
      if (res?.data?.deleteInvitation) {
        toast.success(`${node.email} deleted`)
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
      <CreateInvitationForm onCreated={handleInvitationCreated} />
      <div className="mt-4">
        <LoadingWrapper loading={fetching}>
          <Table className="border-b">
            {!!currentPageInvits?.length && (
              <TableHeader>
                <TableRow>
                  <TableHead className="w-[25%]">Invitee</TableHead>
                  <TableHead className="w-[45%]">Created</TableHead>
                  <TableHead></TableHead>
                </TableRow>
              </TableHeader>
            )}
            <TableBody>
              {currentPageInvits?.map(x => {
                const link = `${externalUrl}/auth/signup?invitationCode=${x.node.code}`
                return (
                  <TableRow key={x.node.id}>
                    <TableCell>{x.node.email}</TableCell>
                    <TableCell>
                      {moment.utc(x.node.createdAt).fromNow()}
                    </TableCell>
                    <TableCell className="flex justify-end">
                      <div className="flex gap-1">
                        <CopyButton value={link} />
                        <Button
                          size="icon"
                          variant="hover-destructive"
                          onClick={() => handleDeleteInvatation(x.node)}
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
        </LoadingWrapper>
      </div>
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
    </div>
  )
}

function getPageNumber(count?: number) {
  return Math.ceil((count || 0) / PAGE_SIZE)
}
