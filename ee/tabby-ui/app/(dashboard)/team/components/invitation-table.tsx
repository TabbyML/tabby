'use client'

import React, { useEffect, useState } from 'react'
import moment from 'moment'
import { useQuery } from 'urql'

import { graphql } from '@/lib/gql/generates'
import { QueryVariables, useMutation } from '@/lib/tabby/gql'
import { Button } from '@/components/ui/button'
import { IconTrash } from '@/components/ui/icons'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow
} from '@/components/ui/table'
import { CopyButton } from '@/components/copy-button'

import CreateInvitationForm from './create-invitation-form'
import { ListInvitationsQuery } from '@/lib/gql/generates/graphql'
import { Pagination, PaginationContent, PaginationItem, PaginationNext, PaginationPrevious } from '@/components/ui/pagination'
import { toast } from 'sonner'

const listInvitations = graphql(/* GraphQL */ `
  query ListInvitations(
    $after: String
    $before: String
    $first: Int
    $last: Int
  ) {
    invitationsNext(
      after: $after
      before: $before
      first: $first
      last: $last
    ) {
      edges {
        node {
          id
          email
          code
          createdAt
        }
        cursor
      }
      pageInfo {
        hasNextPage
        hasPreviousPage
        startCursor
        endCursor
      }
    }
  }
`)

const deleteInvitationMutation = graphql(/* GraphQL */ `
  mutation DeleteInvitation($id: ID!) {
    deleteInvitationNext(id: $id)
  }
`)


const PAGE_SIZE = 5
export default function InvitationTable() {
  const [queryVariables, setQueryVariables] =
    React.useState<QueryVariables<typeof listInvitations>>({
      last: PAGE_SIZE,
    })
  const [invatation, setInvatation] = React.useState<ListInvitationsQuery['invitationsNext']>()
  const [{ data, fetching }, reexecuteQuery] = useQuery({
    query: listInvitations,
    variables: queryVariables
  })

  const updateQueryVariables = (v: typeof queryVariables) => {
    setQueryVariables({
      ...v,
      // urql performs a shallow comparison on variables, adding a random value ensures that a request will be fired
      // @ts-ignore
      randomValue: Math.random()
    })
  }

  const [origin, setOrigin] = useState('')
  useEffect(() => {
    setOrigin(new URL(window.location.href).origin)
  }, [])

  useEffect(() => {
    const _invitations = data?.invitationsNext
    if (_invitations?.edges?.length) {
      setInvatation({
        edges: _invitations.edges.reverse(),
        pageInfo: _invitations.pageInfo
      })
    }
  }, [data])

  const deleteInvitation = useMutation(deleteInvitationMutation)

  const handleInvitationCreated = () => {
    toast.success('Invitation created')
    updateQueryVariables({ last: PAGE_SIZE })
    // if (queryVariables?.after || queryVariables.before) {
    //   updateQueryVariables({ last: PAGE_SIZE })
    // } else {
    //   reexecuteQuery()
    // }
  }

  const handleFetchPrevPage = () => {
    if (fetching) return
    updateQueryVariables({
      first: PAGE_SIZE,
      after: pageInfo?.endCursor
    })
  }

  const handleFetchNextPage = () => {
    if (fetching) return
    updateQueryVariables({
      last: PAGE_SIZE,
      before: pageInfo?.startCursor
    })
  }

  const handleDeleteInvatation = (id: string) => {
    deleteInvitation({ id }).then(() => {
      // due to the `last` direction
      // hasNextPage means that current page is not the first page
      // if there is only one record in current page, while redirect to the first page
      if (pageInfo?.hasNextPage && invatation?.edges?.length !== 1) {
        reexecuteQuery()
      } else {
        updateQueryVariables({ last: PAGE_SIZE })
      }
    })
  }

  const invitations = invatation?.edges
  const pageInfo = invatation?.pageInfo

  return (
    <div>
      <CreateInvitationForm onCreated={handleInvitationCreated} />
      <Table className="border-b mt-4">
        {!!invitations?.length && (
          <TableHeader>
            <TableRow>
              <TableHead className="w-[25%]">Invitee</TableHead>
              <TableHead className="w-[45%]">Created</TableHead>
              <TableHead></TableHead>
            </TableRow>
          </TableHeader>
        )}
        <TableBody>
          {invitations?.map(x => {
            const link = `${origin}/auth/signup?invitationCode=${x.node.code}`
            return (
              <TableRow key={x.node.id}>
                <TableCell>{x.node.email}</TableCell>
                <TableCell>{moment.utc(x.node.createdAt).fromNow()}</TableCell>
                <TableCell className="flex justify-end">
                  <div className="flex gap-1">
                    <CopyButton value={link} />
                    <Button
                      size="icon"
                      variant="hover-destructive"
                      onClick={() => handleDeleteInvatation(x.node.id)}
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
      {(pageInfo?.hasNextPage || pageInfo?.hasPreviousPage) && (
        <Pagination className="my-4">
          <PaginationContent>
            <PaginationItem>
              <PaginationPrevious
                disabled={!pageInfo?.hasNextPage}
                onClick={handleFetchPrevPage}
              />
            </PaginationItem>
            <PaginationItem>
              <PaginationNext
                disabled={!pageInfo?.hasPreviousPage}
                onClick={handleFetchNextPage}
              />
            </PaginationItem>
          </PaginationContent>
        </Pagination>
      )}
    </div>
  )
}
