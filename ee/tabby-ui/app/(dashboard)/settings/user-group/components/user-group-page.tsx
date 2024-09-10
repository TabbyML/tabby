'use client'

import React from 'react'
import { useRouter } from 'next/navigation'
import { toast } from 'sonner'
import { useQuery } from 'urql'

import { DEFAULT_PAGE_SIZE } from '@/lib/constants'
import { graphql } from '@/lib/gql/generates'
import type { ListUsersQuery } from '@/lib/gql/generates/graphql'
import { useMe } from '@/lib/hooks/use-me'
import { QueryVariables, useMutation } from '@/lib/tabby/gql'
import { listUsers } from '@/lib/tabby/query'
import type { ArrayElementType } from '@/lib/types'
import { Button } from '@/components/ui/button'
import LoadingWrapper from '@/components/loading-wrapper'

import CreateUserGroupDialog from './create-user-group'
import { UserGroupItem } from './user-group-item'

const updateUserActiveMutation = graphql(/* GraphQL */ `
  mutation UpdateUserActive($id: ID!, $active: Boolean!) {
    updateUserActive(id: $id, active: $active)
  }
`)

type UserNode = ArrayElementType<ListUsersQuery['users']['edges']>['node']

const PAGE_SIZE = DEFAULT_PAGE_SIZE
export default function UsersTable() {
  const router = useRouter()
  const [{ data: me }] = useMe()
  const [queryVariables, setQueryVariables] = React.useState<
    QueryVariables<typeof listUsers>
  >({ first: PAGE_SIZE })
  const [{ data, error, fetching }, reexecuteQuery] = useQuery({
    query: listUsers,
    variables: queryVariables
  })
  const [users, setUsers] = React.useState<ListUsersQuery['users']>()

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
    })
  }

  const onCreateUserGroup = async () => {
    // console.log('submit')
    // refetch list
  }

  const pageInfo = users?.pageInfo

  return (
    <LoadingWrapper loading={fetching}>
      <div className="flex justify-end mb-4">
        <CreateUserGroupDialog onSubmit={onCreateUserGroup}>
          <Button type="button">Create</Button>
        </CreateUserGroupDialog>
      </div>
      {!!users?.edges?.length && (
        <div className="border border-b-0">
          {users.edges.map(group => {
            // FIXME
            return (
              <UserGroupItem key={group.node.id} userGroup={group as any} />
            )
          })}
        </div>
      )}
    </LoadingWrapper>
  )
}
