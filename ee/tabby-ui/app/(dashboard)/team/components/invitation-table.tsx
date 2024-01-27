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
import { SimplePagination } from '@/components/simple-pagination'

import CreateInvitationForm from './create-invitation-form'

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

const PAGE_SIZE = 2
export default function InvitationTable() {
  const [queryVariables, setQueryVariables] = React.useState<
    QueryVariables<typeof listInvitations>
  >({
    first: PAGE_SIZE,
  })
  const [{ data }, reexecuteQuery] = useQuery({
    query: listInvitations,
    variables: queryVariables
  })
  const invitations = data?.invitationsNext?.edges
  const pageInfo = data?.invitationsNext?.pageInfo
  const [origin, setOrigin] = useState('')
  useEffect(() => {
    setOrigin(new URL(window.location.href).origin)
  }, [])

  const deleteInvitation = useMutation(deleteInvitationMutation, {
    onCompleted() {
      reexecuteQuery()
    }
  })

  return (
    <div>
      <Table className="border-b">
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
                <TableCell className="text-center">
                  <CopyButton value={link} />
                  <Button
                    size="icon"
                    variant="hover-destructive"
                    onClick={() => deleteInvitation(x.node)}
                  >
                    <IconTrash />
                  </Button>
                </TableCell>
              </TableRow>
            )
          })}
        </TableBody>
      </Table>
      <div className="flex justify-between mt-4 items-start">
        <CreateInvitationForm onCreated={() => reexecuteQuery()} />
        {!!invitations?.length && (
          <SimplePagination
            hasNextPage={pageInfo?.hasNextPage}
            hasPreviousPage={pageInfo?.hasPreviousPage}
            onNext={() =>
              setQueryVariables({
                first: PAGE_SIZE,
                after: pageInfo?.endCursor
              })
            }
            onPrev={() =>
              setQueryVariables({
                last: PAGE_SIZE,
                before: pageInfo?.startCursor
              })
            }
          />
        )}
      </div>
    </div>
  )
}
