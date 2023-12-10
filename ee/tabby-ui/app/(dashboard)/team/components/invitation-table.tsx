'use client'

import React, { useEffect, useState } from 'react'
import moment from 'moment'

import { graphql } from '@/lib/gql/generates'
import { useAuthenticatedGraphQLQuery, useMutation } from '@/lib/tabby/gql'
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

export default function InvitationTable() {
  const { data, mutate } = useAuthenticatedGraphQLQuery(listInvitations)
  const invitations = data?.invitations
  const [origin, setOrigin] = useState('')
  useEffect(() => {
    setOrigin(new URL(window.location.href).origin)
  }, [])

  const deleteInvitation = useMutation(deleteInvitationMutation, {
    onCompleted() {
      mutate()
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
            const link = `${origin}/auth/signup?invitationCode=${x.code}`
            return (
              <TableRow key={i}>
                <TableCell>{x.email}</TableCell>
                <TableCell>{moment.utc(x.createdAt).fromNow()}</TableCell>
                <TableCell className="text-center">
                  <CopyButton value={link} />
                  <Button
                    size="icon"
                    variant="hover-destructive"
                    onClick={() => deleteInvitation(x)}
                  >
                    <IconTrash />
                  </Button>
                </TableCell>
              </TableRow>
            )
          })}
          <TableRow>
            <TableCell className="p-2">
              <CreateInvitationForm onCreated={() => mutate()} />
            </TableCell>
          </TableRow>
        </TableBody>
      </Table>
    )
  )
}
