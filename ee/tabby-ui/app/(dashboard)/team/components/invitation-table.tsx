'use client'

import React, { useEffect, useState } from 'react'
import moment from 'moment'
import { useQuery } from 'urql'

import { graphql } from '@/lib/gql/generates'
import { useMutation } from '@/lib/tabby/gql'
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

const listInvitations = graphql(/* GraphQL */ `
  query ListInvitations($after: String, $before: String, $first: Int, $last: Int) {
    invitationsNext(after: $after, before: $before, first: $first, last: $last) {
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

export default function InvitationTable() {
  const [{ data }, reexecuteQuery] = useQuery({ query: listInvitations })
  const invitations = data?.invitationsNext?.edges
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
    invitations && (
      <Table>
        {invitations.length > 0 && (
          <TableHeader>
            <TableRow>
              <TableHead className="w-[25%]">Invitee</TableHead>
              <TableHead className="w-[45%]">Created</TableHead>
              <TableHead></TableHead>
            </TableRow>
          </TableHeader>
        )}
        <TableBody>
          {invitations.map((x, i) => {
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
          <TableRow>
            <TableCell className="p-2">
              <CreateInvitationForm onCreated={() => reexecuteQuery()} />
            </TableCell>
          </TableRow>
        </TableBody>
      </Table>
    )
  )
}
