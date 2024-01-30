'use client'

import React from 'react'
import moment from 'moment'
import { toast } from 'sonner'
import { useQuery } from 'urql'

import { graphql } from '@/lib/gql/generates'
import { QueryVariables, useMutation } from '@/lib/tabby/gql'
import type { ArrayElementType } from '@/lib/types'
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

const PAGE_SIZE = 2
export default function UsersTable() {
  const [queryVariables, setQueryVariables] = React.useState<
    QueryVariables<typeof listUsers>
  >({ first: PAGE_SIZE })
  const [{ data }, reexecuteQuery] = useQuery({
    query: listUsers,
    variables: queryVariables
  })
  const users = data?.usersNext?.edges
  const pageInfo = data?.usersNext?.pageInfo

  const updateUserActive = useMutation(updateUserActiveMutation)

  const onUpdateUserActive = (
    node: ArrayElementType<typeof users>['node'],
    active: boolean
  ) => {
    updateUserActive({ id: node.id, active }).then(response => {
      if (response?.error || !response?.data?.updateUserActive) {
        toast.error(
          response?.error?.message ||
            `${active ? 'activate' : 'deactivate'} failed`
        )
        return
      }

      reexecuteQuery()
      toast.success(`${node.email} is ${active ? 'activated' : 'deactivated'}`)
    })
  }

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
                          onSelect={() => onUpdateUserActive(x.node, false)}
                          className="cursor-pointer"
                        >
                          <span className="ml-2">Deactivate</span>
                        </DropdownMenuItem>
                      )}
                      {!x.node.active && (
                        <DropdownMenuItem
                          onSelect={() => onUpdateUserActive(x.node, true)}
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
        <div className="flex justify-end my-4">
          <SimplePagination
            hasPreviousPage={pageInfo?.hasPreviousPage}
            hasNextPage={pageInfo?.hasNextPage}
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
        </div>
      </>
    )
  )
}
