'use client'

import React from 'react'
import moment from 'moment'

import { graphql } from '@/lib/gql/generates'
import { useAuthenticatedGraphQLQuery } from '@/lib/tabby/gql'
import { Badge } from '@/components/ui/badge'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'

const listUsers = graphql(/* GraphQL */ `
  query ListUsers {
    users {
      email
      isAdmin
      createdAt
    }
  }
`)

export default function UsersTable() {
  const { data } = useAuthenticatedGraphQLQuery(listUsers)
  const users = data?.users

  return (
    users && (
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead className="w-[25%]">Email</TableHead>
            <TableHead className="w-[45%]">Joined</TableHead>
            <TableHead>Role</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {users.map((x, i) => (
            <TableRow key={i}>
              <TableCell>{x.email}</TableCell>
              <TableCell>{moment.utc(x.createdAt).fromNow()}</TableCell>
              <TableCell>
                {x.isAdmin ? (
                  <Badge>OWNER</Badge>
                ) : (
                  <Badge variant="secondary">MEMBER</Badge>
                )}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    )
  )
}
