import { on } from 'events'
import { HTMLAttributes, useContext } from 'react'
import { toast } from 'sonner'
import { useQuery } from 'urql'

import { graphql } from '@/lib/gql/generates'
import { UserGroupMembership } from '@/lib/gql/generates/graphql'
import { useMutation } from '@/lib/tabby/gql'
import { cn } from '@/lib/utils'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { IconEdit, IconPlus, IconTrash, IconUser } from '@/components/ui/icons'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Skeleton } from '@/components/ui/skeleton'
import LoadingWrapper from '@/components/loading-wrapper'

import UpsertMemberDialog from './upsert-member-dialog'
import { UserGroupContext } from './user-group-page'

const deleteUserGroupMembershipMutation = graphql(/* GraphQL */ `
  mutation DeleteUserGroupMembership($userGroupId: ID!, $userId: ID!) {
    deleteUserGroupMembership(userGroupId: $userGroupId, userId: $userId)
  }
`)

interface MembershipViewProps extends HTMLAttributes<HTMLDivElement> {
  userGroupId: string
  userGroupName: string
  members: UserGroupMembership[]
  onUpdate: () => void
  editable: boolean
}

export function MembershipView({
  userGroupId,
  userGroupName,
  className,
  members,
  onUpdate,
  editable
}: MembershipViewProps) {
  const { allUsers, fetchingAllUsers } = useContext(UserGroupContext)
  const deleteUserGroupMembership = useMutation(
    deleteUserGroupMembershipMutation
  )

  const handleDeleteUserGroupMembership = (userId: string) => {
    return deleteUserGroupMembership({
      userGroupId,
      userId
    })
      .then(res => {
        if (!res?.data?.deleteUserGroupMembership) {
          const errorMessage = res?.error?.message || 'Failed to delete'
          toast.error(errorMessage)
          return
        }

        onUpdate()
      })
      .catch(error => {
        toast.error(error.message || 'Failed to delete')
      })
  }

  const existingMemberIds = members.map(o => o.user.id)

  return (
    <div
      className={cn(
        'max-h-[300px] flex-col gap-1 overflow-hidden border-b',
        className
      )}
    >
      <ScrollArea className="flex-1">
        {members?.length ? (
          members.map(item => {
            const member = allUsers.find(o => o.id === item.user.id)
            return (
              <div
                key={item.user.id}
                className="flex items-center gap-2 border-b py-3 pl-10 pr-3 hover:bg-muted/50"
              >
                <IconUser className="shrink-0" />
                <div className="flex-1">
                  <div className="flex items-center gap-2 text-sm">
                    <LoadingWrapper
                      loading={fetchingAllUsers}
                      fallback={<Skeleton className="w-14" />}
                    >
                      <span>{member?.name ?? ''}</span>
                    </LoadingWrapper>
                    {item.isGroupAdmin ? (
                      <Badge>Group Admin</Badge>
                    ) : (
                      <Badge variant="secondary">Group Member</Badge>
                    )}
                  </div>
                  <LoadingWrapper
                    loading={fetchingAllUsers}
                    fallback={<Skeleton className="w-14" />}
                  >
                    <div className="text-sm text-muted-foreground">
                      {member?.email}
                    </div>
                  </LoadingWrapper>
                </div>
                <div className="flex items-center gap-2">
                  <UpsertMemberDialog
                    isNew={false}
                    userGroupId={userGroupId}
                    userGroupName={userGroupName}
                    onSuccess={onUpdate}
                    existingMemberIds={existingMemberIds}
                    initialValues={item}
                  >
                    <Button className="shrink-0" variant="ghost" size="icon">
                      <IconEdit />
                    </Button>
                  </UpsertMemberDialog>
                  <Button
                    className="shrink-0"
                    variant="hover-destructive"
                    size="icon"
                    onClick={e => handleDeleteUserGroupMembership(item.user.id)}
                  >
                    <IconTrash />
                  </Button>
                </div>
              </div>
            )
          })
        ) : (
          <div className="px-3 py-4 text-center">No members</div>
        )}
      </ScrollArea>
      {editable && (
        <div className="mb-8 mt-4 flex justify-center">
          <UpsertMemberDialog
            isNew
            userGroupId={userGroupId}
            userGroupName={userGroupName}
            onSuccess={onUpdate}
            existingMemberIds={existingMemberIds}
          >
            <Button>
              <IconPlus />
              Add Member
            </Button>
          </UpsertMemberDialog>
        </div>
      )}
    </div>
  )
}
