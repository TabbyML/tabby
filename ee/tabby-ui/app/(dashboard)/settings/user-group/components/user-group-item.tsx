import { HTMLAttributes, useState } from 'react'

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

interface UserGroupItemProps extends HTMLAttributes<HTMLDivElement> {
  // todo type
  userGroup: any
}

export function UserGroupItem({ userGroup, className }: UserGroupItemProps) {
  const [deleteAlertVisible, setDeleteAlertVisible] = useState(false)
  const [membershipOpen, setMembershipOpen] = useState(false)
  const [isDeleting, setIsDeleting] = useState(false)
  const toggleMembership = () => setMembershipOpen(!membershipOpen)

  const onDelete: React.MouseEventHandler<HTMLButtonElement> = e => {
    e.preventDefault()
    setIsDeleting(true)
    // todo delete user group
    // deleteOAuthCredential({ provider: providerValue }).then(res => {
    //   if (res?.data?.deleteOauthCredential) {
    //     navigateToSSOSettings()
    //   } else {
    //     setIsDeleting(false)
    //     if (res?.error) {
    //       toast.error(res?.error?.message)
    //     }
    //   }
    // })
  }

  return (
    <div>
      <div
        className={cn(
          'flex items-center gap-2 p-3 hover:bg-muted/50 cursor-pointer border-b',
          className
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
        <div className="flex-1 overflow-hidden">Group Name</div>
        <div
          onClick={e => e.stopPropagation()}
          className="flex items-center gap-2"
        >
          {/* <Button size='icon' variant='ghost'>
            <IconPlus />
          </Button> */}
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
                  onClick={onDelete}
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
      {membershipOpen && <MembershipView userGroupId="1" />}
    </div>
  )
}
