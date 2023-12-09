'use client'

import React, { useImperativeHandle } from 'react'
import moment from 'moment'

import { graphql } from '@/lib/gql/generates'
import { useAuthenticatedGraphQLQuery, useGraphQLForm } from '@/lib/tabby/gql'
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

const listInvitations = graphql(/* GraphQL */ `
  query ListInvitations {
    invitations {
      id
      email
      code
      createdAt
    }
  }
`)

const deleteInvitationMutation = graphql(/* GraphQL */ `
  mutation DeleteInvitation($id: Int!) {
    deleteInvitation(id: $id)
  }
`)

type InvitationTableHandle = {
  changed: () => void
}

const InvitationTable: React.ForwardRefRenderFunction<
  InvitationTableHandle,
  {}
> = (_, forwardedRef) => {
  const { data, mutate } = useAuthenticatedGraphQLQuery(listInvitations)
  const invitations = data?.invitations

  const url = new URL(window.location.href)
  const { onSubmit: deleteInvitation } = useGraphQLForm(
    deleteInvitationMutation,
    {
      onSuccess: () => mutate()
    }
  )

  useImperativeHandle(
    forwardedRef,
    () => {
      return {
        changed() {
          mutate()
        }
      }
    },
    [mutate]
  )

  return (
    invitations && (
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Invitee</TableHead>
            <TableHead>Created</TableHead>
            <TableHead></TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {invitations.map((x, i) => {
            const link = `${url.origin}/auth/signup?invitationCode=${x.code}`
            return (
              <TableRow key={i}>
                <TableCell className="font-medium">{x.email}</TableCell>
                <TableCell>{moment(x.createdAt).fromNow()}</TableCell>
                <TableCell className="flex items-center">
                  <CopyButton value={link} />
                  <Button
                    size="icon"
                    variant="hover-destructive"
                    onClick={() => deleteInvitation({ id: x.id })}
                  >
                    <IconTrash />
                  </Button>
                </TableCell>
              </TableRow>
            )
          })}
        </TableBody>
      </Table>
    )
  )
}

export default React.forwardRef(InvitationTable)
