'use client'

import React from 'react'
import moment from 'moment'
import { useQuery } from 'urql'

import { graphql } from '@/lib/gql/generates'
import { QueryVariables, useMutation } from '@/lib/tabby/gql'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import { IconMore } from '@/components/ui/icons'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow
} from '@/components/ui/table'
import { toast } from 'sonner'

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
          active
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

const updateUserActiveMutation = graphql(/* GraphQL */ `
  mutation UpdateUserActive($id: ID!, $active: Boolean!) {
    updateUserActive(id: $id, active: $active)
  }
`)

export default function UsersTable() {
  const [queryVariables, setQueryVariables] =
    React.useState<QueryVariables<typeof listUsers>>()
  const [{ data }, reexecuteQuery] = useQuery({
    query: listUsers,
    variables: queryVariables
  })
  const users = data?.usersNext?.edges

  const updateUserActive = useMutation(updateUserActiveMutation, {
    onCompleted(values) {
      if (values?.updateUserActive) {
        toast.success('success')
        reexecuteQuery()
      }
    },
    onError: (e) => {
      toast.error(e.message)
    }
  })

  return (
    !!users?.length && (
      <>
        <Table className="border-b">
          <TableHeader>
            <TableRow>
              <TableHead className="w-[25%]">Email</TableHead>
              <TableHead className="w-[35%]">Joined</TableHead>
              <TableHead className="w-[15%] text-center">Status</TableHead>
              <TableHead className="w-[15%] text-center">Level</TableHead>
              <TableHead className="w-[100px]"></TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {users.map(x => (
              <TableRow key={x.node.id}>
                <TableCell>{x.node.email}</TableCell>
                <TableCell>{moment.utc(x.node.createdAt).fromNow()}</TableCell>
                <TableCell className="text-center">
                  {x.node.active ? (
                    <Badge variant="successful">Active</Badge>
                  ) : (
                    <Badge variant="secondary">Inactive</Badge>
                  )}
                </TableCell>
                <TableCell className="text-center">
                  {x.node.isAdmin ? (
                    <Badge>ADMIN</Badge>
                  ) : (
                    <Badge variant="secondary">MEMBER</Badge>
                  )}
                </TableCell>
                <TableCell className="flex justify-end">
                  <DropdownMenu>
                    <DropdownMenuTrigger>
                      {x.node.isAdmin ? null : (
                        <Button size="icon" variant="ghost">
                          <IconMore />
                        </Button>
                      )}
                    </DropdownMenuTrigger>
                    <DropdownMenuContent collisionPadding={{ right: 16 }}>
                      {x.node.active && (
                        <DropdownMenuItem
                          onSelect={() =>
                            updateUserActive({
                              id: x.node.id,
                              active: false
                            })
                          }
                          className="cursor-pointer"
                        >
                          <span className="ml-2">Deactivate</span>
                        </DropdownMenuItem>
                      )}
                      {!x.node.active && (
                        <DropdownMenuItem
                          onSelect={() =>
                            updateUserActive({
                              id: x.node.id,
                              active: true
                            })
                          }
                          className="cursor-pointer"
                        >
                          <span className="ml-2">Activate</span>
                        </DropdownMenuItem>
                      )}
                    </DropdownMenuContent>
                  </DropdownMenu>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </>
    )
  )
}
