'use client'

import React from 'react'
import { toast } from 'sonner'
import { useQuery } from 'urql'

import { userGroupsQuery } from '@/lib/tabby/query'
import { Button } from '@/components/ui/button'
import LoadingWrapper from '@/components/loading-wrapper'
import { ListSkeleton } from '@/components/skeleton'

import CreateUserGroupDialog from './create-user-group'
import { UserGroupItem } from './user-group-item'

export default function UsersTable() {
  const [{ data, error, fetching }, reexcute] = useQuery({
    query: userGroupsQuery
  })

  React.useEffect(() => {
    if (error?.message) {
      toast.error(error.message)
    }
  }, [error])

  const onCreateUserGroup = async () => {
    // console.log('submit')
    reexcute()
  }

  const userGroups = data?.userGroups

  return (
    <LoadingWrapper
      loading={fetching}
      fallback={<ListSkeleton className="mt-12" />}
    >
      <div className="flex justify-end mb-4">
        <CreateUserGroupDialog onSubmit={onCreateUserGroup}>
          <Button type="button">Create</Button>
        </CreateUserGroupDialog>
      </div>
      {userGroups?.length ? (
        <div className="border rounded-lg overflow-hidden">
          <div className="border-b bg-muted font-semibold py-3 px-4">
            Groups
          </div>
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
        </div>
      )}
    </LoadingWrapper>
  )
}
