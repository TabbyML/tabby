'use client'

import React from 'react'
import moment from 'moment'

import { graphql } from '@/lib/gql/generates'
import { useAuthenticatedGraphQLQuery } from '@/lib/tabby/gql'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow
} from '@/components/ui/table'

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
  const { data, mutate } = useAuthenticatedGraphQLQuery(listUsers)
  const users = data?.users

  return (
    users && (
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Email</TableHead>
            <TableHead>Joined</TableHead>
            <TableHead>Role</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {users.map((x, i) => (
            <TableRow key={i}>
              <TableCell className="w-[300px] font-medium">{x.email}</TableCell>
              <TableCell>{moment.utc(x.createdAt).fromNow()}</TableCell>
              <TableCell>{x.isAdmin ? 'Admin' : 'Member'}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    )
  )
}
