import { useState } from 'react'

import { UserGroupMembership } from '@/lib/gql/generates/graphql'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger
} from '@/components/ui/dialog'

import UpsertMembershipForm from './upsert-member-form'

interface UpsertMemberDialogProps {
  userGroupId: string
  userGroupName: string
  isNew: boolean
  children: React.ReactNode
  onSuccess?: () => void
  existingMemberIds?: string[]
  initialValues?: UserGroupMembership
}

export default function UpsertMemberDialog({
  children,
  userGroupId,
  isNew,
  userGroupName,
  onSuccess,
  existingMemberIds,
  initialValues
}: UpsertMemberDialogProps) {
  const [open, setOpen] = useState(false)

  const handleOpenChange = (open: boolean) => {
    setOpen(open)
  }

  const handleSuccess = () => {
    onSuccess?.()
    handleOpenChange(false)
  }

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent>
        <DialogHeader className="gap-3">
          <DialogTitle>{isNew ? 'Add member' : 'Update member'}</DialogTitle>
        </DialogHeader>
        <UpsertMembershipForm
          onCancel={() => handleOpenChange(false)}
          onSuccess={handleSuccess}
          userGroupId={userGroupId}
          isNew={isNew}
          existingMemberIds={existingMemberIds}
          initialValues={initialValues}
        />
      </DialogContent>
      <DialogTrigger asChild>{children}</DialogTrigger>
    </Dialog>
  )
}
