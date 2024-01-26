'use client'

import React from 'react'
import moment from 'moment'
import { useQuery } from 'urql'

import { graphql } from '@/lib/gql/generates'
import { Badge } from '@/components/ui/badge'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow
} from '@/components/ui/table'

const listUsers = graphql(/* GraphQL */ `
  query ListUsersNext($after: String, $before: String, $first: Int, $last: Int) {
    usersNext(after: $after, before: $before, first: $first, last: $last) {
      edges {
        node {
          email
          isAdmin
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

export default function UsersTable() {
  const [{ data }] = useQuery({ query: listUsers })
  const users = data?.usersNext?.edges

  return (
    users && (
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead className="w-[25%]">Email</TableHead>
            <TableHead className="w-[45%]">Joined</TableHead>
            <TableHead className="text-center">Level</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {users.map((x, i) => (
            <TableRow key={x.node.email}>
              <TableCell>{x.node.email}</TableCell>
              <TableCell>{moment.utc(x.node.createdAt).fromNow()}</TableCell>
              <TableCell className="text-center">
                {x.node.isAdmin ? (
                  <Badge>ADMIN</Badge>
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
