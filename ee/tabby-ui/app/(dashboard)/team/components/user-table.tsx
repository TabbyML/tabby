'use client'

import React from 'react'
import moment from 'moment'
import { useQuery } from 'urql'

import { graphql } from '@/lib/gql/generates'
import { QueryVariables } from '@/lib/tabby/gql'
import { Badge } from '@/components/ui/badge'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow
} from '@/components/ui/table'
import { SimplePagination } from '@/components/simple-pagination'

const listUsers = graphql(/* GraphQL */ `
  query ListUsersNext(
    $after: String
    $before: String
    $first: Int
    $last: Int
  ) {
    usersNext(after: $after, before: $before, first: $first, last: $last) {
      edges {
        node {
          id
          email
          isAdmin
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

const PAGE_SIZE = 5
export default function UsersTable() {
  const [queryVariables, setQueryVariables] = React.useState<
    QueryVariables<typeof listUsers>
  >({
    first: PAGE_SIZE
  })
  const [{ data }] = useQuery({
    query: listUsers,
    variables: queryVariables
  })
  const users = data?.usersNext?.edges
  const pageInfo = data?.usersNext?.pageInfo

  return (
    !!users?.length && (
      <>
        <Table className="border-b">
          <TableHeader>
            <TableRow>
              <TableHead className="w-[25%]">Email</TableHead>
              <TableHead className="w-[45%]">Joined</TableHead>
              <TableHead className="text-center">Level</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {users.map(x => (
              <TableRow key={x.node.id}>
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
        <div className="flex justify-end mt-4">
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
                first: PAGE_SIZE,
                before: pageInfo?.startCursor
              })
            }
          />
        </div>
      </>
    )
  )
}
