'use client'

import React from 'react'
import moment from 'moment'
import { toast } from 'sonner'
import { useQuery } from 'urql'

import { DEFAULT_PAGE_SIZE } from '@/lib/constants'
import { graphql } from '@/lib/gql/generates'
import type { ListUsersQuery } from '@/lib/gql/generates/graphql'
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
  Pagination,
  PaginationContent,
  PaginationItem,
  PaginationNext,
  PaginationPrevious
} from '@/components/ui/pagination'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow
} from '@/components/ui/table'

import { UpdateUserRoleDialog } from './user-role-dialog'

const listUsers = graphql(/* GraphQL */ `
  query ListUsers($after: String, $before: String, $first: Int, $last: Int) {
    users(after: $after, before: $before, first: $first, last: $last) {
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

type UserNode = ArrayElementType<ListUsersQuery['users']['edges']>['node']

const PAGE_SIZE = DEFAULT_PAGE_SIZE
export default function UsersTable() {
  const [queryVariables, setQueryVariables] = React.useState<
    QueryVariables<typeof listUsers>
  >({ first: PAGE_SIZE })
  const [{ data, error }, reexecuteQuery] = useQuery({
    query: listUsers,
    variables: queryVariables
  })
  const [users, setUsers] = React.useState<ListUsersQuery['users']>()
  const [currentUser, setCurrentUser] = React.useState<UserNode>()
  const [updateRoleVisible, setUpdateRoleVisible] = React.useState(false)

  React.useEffect(() => {
    const _users = data?.users
    if (_users?.edges?.length) {
      setUsers(_users)
    }
  }, [data])

  React.useEffect(() => {
    if (error?.message) {
      toast.error(error.message)
    }
  }, [error])

  const updateUserActive = useMutation(updateUserActiveMutation)

  const onUpdateUserActive = (node: UserNode, active: boolean) => {
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

  const onUpdateUserRole = (node: UserNode) => {
    setCurrentUser(node)
    setUpdateRoleVisible(true)
  }

  const pageInfo = users?.pageInfo

  return (
    !!users?.edges?.length && (
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
            {users.edges.map(x => (
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
                <TableCell className="text-end">
                  <DropdownMenu modal={false}>
                    <DropdownMenuTrigger asChild>
                      <div className="h-8">
                        <Button size="icon" variant="ghost">
                          <IconMore />
                        </Button>
                      </div>
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
                      <DropdownMenuItem
                        onSelect={() => onUpdateUserRole(x.node)}
                        className="cursor-pointer"
                      >
                        <span className="ml-2">Update Role</span>
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
        {(pageInfo?.hasNextPage || pageInfo?.hasPreviousPage) && (
          <Pagination className="my-4">
            <PaginationContent>
              <PaginationItem>
                <PaginationPrevious
                  disabled={!pageInfo?.hasPreviousPage}
                  onClick={e =>
                    setQueryVariables({
                      last: PAGE_SIZE,
                      before: pageInfo?.startCursor
                    })
                  }
                />
              </PaginationItem>
              <PaginationItem>
                <PaginationNext
                  disabled={!pageInfo?.hasNextPage}
                  onClick={e =>
                    setQueryVariables({
                      first: PAGE_SIZE,
                      after: pageInfo?.endCursor
                    })
                  }
                />
              </PaginationItem>
            </PaginationContent>
          </Pagination>
        )}

        <UpdateUserRoleDialog
          onSuccess={() => {
            reexecuteQuery()
            setUpdateRoleVisible(false)
          }}
          user={currentUser}
          open={updateRoleVisible}
          onOpenChange={setUpdateRoleVisible}
        />
      </>
    )
  )
}
