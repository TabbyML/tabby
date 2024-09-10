import { HTMLAttributes, useState } from 'react'
import { toast } from 'sonner'

import { graphql } from '@/lib/gql/generates'
import { UserGroup } from '@/lib/gql/generates/graphql'
import { useMutation } from '@/lib/tabby/gql'
import { cn } from '@/lib/utils'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger
} from '@/components/ui/alert-dialog'
import { Button, buttonVariants } from '@/components/ui/button'
import {
  IconChevronRight,
  IconPlus,
  IconSpinner,
  IconTrash,
  IconUsers
} from '@/components/ui/icons'

import { MembershipView } from './membership-view'

const deleteUserGroupMutation = graphql(/* GraphQL */ `
  mutation DeleteUserGroup($id: ID!) {
    deleteUserGroup(id: $id)
  }
`)

interface UserGroupItemProps extends HTMLAttributes<HTMLDivElement> {
  userGroup: UserGroup
  onSuccess?: () => void
  isLastItem?: boolean
}

export function UserGroupItem({
  onSuccess,
  userGroup,
  isLastItem
}: UserGroupItemProps) {
  const [deleteAlertVisible, setDeleteAlertVisible] = useState(false)
  const [membershipOpen, setMembershipOpen] = useState(false)
  const [isDeleting, setIsDeleting] = useState(false)
  const toggleMembership = () => setMembershipOpen(!membershipOpen)

  const deleteUserGroup = useMutation(deleteUserGroupMutation)

  const handleDeleteUserGroup: React.MouseEventHandler<
    HTMLButtonElement
  > = e => {
    e.preventDefault()
    setIsDeleting(true)

    deleteUserGroup({
      id: userGroup.id
    })
      .then(res => {
        if (!res?.data?.deleteUserGroup) {
          const errorMessage = res?.error?.message || 'Failed to delete'
          // todo show errorMsg in dialog content
          toast.error(errorMessage)
        } else {
          onSuccess?.()
          setMembershipOpen(false)
        }
      })
      .catch(e => {
        const errorMessage = e?.message || 'Failed to delete'
        toast.error(errorMessage)
      })
      .finally(() => {
        setIsDeleting(false)
      })
  }

  return (
    <div>
      <div
        className={cn(
          'flex items-center gap-2 p-3 hover:bg-muted/50 cursor-pointer border-b',
          {
            'border-b-0': !!isLastItem && !membershipOpen
          }
        )}
        onClick={toggleMembership}
        // onDoubleClick={e => e.preventDefault()}
      >
        <IconChevronRight
          className={cn('shrink-0 transition-all', {
            'rotate-90': membershipOpen
          })}
        />
        <IconUsers className="shrink-0" />
        <div className="flex-1 overflow-hidden font-semibold">
          {userGroup.name}
        </div>
        <div
          onClick={e => e.stopPropagation()}
          className="flex items-center gap-2"
        >
          <AlertDialog
            open={deleteAlertVisible}
            onOpenChange={setDeleteAlertVisible}
          >
            <AlertDialogTrigger asChild>
              <Button size="icon" variant="hover-destructive">
                <IconTrash />
              </Button>
            </AlertDialogTrigger>
            <AlertDialogContent>
              <AlertDialogHeader>
                <AlertDialogTitle>Are you absolutely sure?</AlertDialogTitle>
                <AlertDialogDescription>
                  It will permanently delete
                  <span className="ml-1 font-bold">Group Name</span>.
                </AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel>Cancel</AlertDialogCancel>
                <AlertDialogAction
                  className={buttonVariants({ variant: 'destructive' })}
                  onClick={handleDeleteUserGroup}
                >
                  {isDeleting && (
                    <IconSpinner className="mr-2 h-4 w-4 animate-spin" />
                  )}
                  Yes, delete it
                </AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>
        </div>
      </div>
      {membershipOpen && (
        <MembershipView
          userGroupId={userGroup.id}
          userGroupName={userGroup.name}
          className={isLastItem ? 'border-b-0' : undefined}
        />
      )}
    </div>
  )
}
