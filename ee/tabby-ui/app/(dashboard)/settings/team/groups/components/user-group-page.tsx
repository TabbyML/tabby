'use client'

import React from 'react'
import { toast } from 'sonner'
import { useQuery } from 'urql'

import { MeQueryQuery } from '@/lib/gql/generates/graphql'
import { Member, useAllMembers } from '@/lib/hooks/use-all-members'
import { useMe } from '@/lib/hooks/use-me'
import { userGroupsQuery } from '@/lib/tabby/query'
import { Button } from '@/components/ui/button'
import { CardHeader, CardTitle } from '@/components/ui/card'
import { IconPlus } from '@/components/ui/icons'
import LoadingWrapper from '@/components/loading-wrapper'
import { ListSkeleton } from '@/components/skeleton'

import CreateUserGroupDialog from './create-user-group'
import { UserGroupItem } from './user-group-item'

interface UserGroupContextValue {
  fetchingAllUsers: boolean
  allUsers: Member[]
  me: MeQueryQuery['me'] | undefined
  refreshUserGroups: () => void
}

export const UserGroupContext = React.createContext<UserGroupContextValue>(
  {} as UserGroupContextValue
)

export default function UserGroups() {
  const [allUsers, fetchingAllUsers] = useAllMembers()
  const [{ data, error, fetching }, reexcute] = useQuery({
    query: userGroupsQuery
  })

  const [{ data: meData }] = useMe()
  const isAdmin = !!(meData?.me.isOwner || meData?.me.isAdmin)
  React.useEffect(() => {
    if (error?.message) {
      toast.error(error.message)
    }
  }, [error])

  const onCreateUserGroup = async () => {
    reexcute()
  }
  const userGroups = data?.userGroups
  return (
    <UserGroupContext.Provider
      value={{
        allUsers,
        fetchingAllUsers,
        refreshUserGroups: reexcute,
        me: meData?.me
      }}
    >
      <CardHeader className="px-0 pt-0">
        <CardTitle className="flex min-h-[2rem] items-start justify-between">
          Groups
          {meData?.me.isAdmin && (
            <CreateUserGroupDialog onSubmit={onCreateUserGroup}>
              <Button className="gap-1">
                <IconPlus />
                Create
              </Button>
            </CreateUserGroupDialog>
          )}
        </CardTitle>
      </CardHeader>
      <LoadingWrapper loading={fetching} fallback={<ListSkeleton />}>
        {userGroups?.length ? (
          <div className="overflow-hidden rounded-lg border">
            {userGroups.map((group, idx) => {
              return (
                <UserGroupItem
                  key={group.id}
                  userGroup={group}
                  onSuccess={() => reexcute()}
                  isLastItem={idx === userGroups.length - 1}
                />
              )
            })}
          </div>
        ) : (
          <div className="flex flex-col items-center gap-4 rounded-lg border-4 border-dashed py-8">
            <div>No Data</div>
            {isAdmin && (
              <div className="mb-4 flex justify-end">
                <CreateUserGroupDialog onSubmit={onCreateUserGroup}>
                  <Button type="button">Create</Button>
                </CreateUserGroupDialog>
              </div>
            )}
          </div>
        )}
      </LoadingWrapper>
    </UserGroupContext.Provider>
  )
}
