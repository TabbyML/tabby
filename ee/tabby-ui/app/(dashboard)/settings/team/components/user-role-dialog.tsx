'use client'

import React from 'react'
import { toast } from 'sonner'

import { graphql } from '@/lib/gql/generates/gql'
import { useMutation } from '@/lib/tabby/gql'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle
} from '@/components/ui/alert-dialog'
import { buttonVariants } from '@/components/ui/button'
import { IconSpinner } from '@/components/ui/icons'

const updateUserRoleMutation = graphql(/* GraphQL */ `
  mutation updateUserRole($id: ID!, $isAdmin: Boolean!) {
    updateUserRole(id: $id, isAdmin: $isAdmin)
  }
`)

interface UpdateUserRoleDialogProps {
  open?: boolean
  onOpenChange?(open: boolean): void
  user?: { id: string; email: string }
  isPromote?: boolean
  onSuccess?: () => void
}

export const UpdateUserRoleDialog: React.FC<UpdateUserRoleDialogProps> = ({
  user,
  onSuccess,
  open,
  onOpenChange,
  isPromote
}) => {
  const [isSubmitting, setIsSubmitting] = React.useState(false)
  const requestPasswordResetEmail = useMutation(updateUserRoleMutation)

  const onSubmit: React.MouseEventHandler<HTMLButtonElement> = async e => {
    e.preventDefault()

    if (!user?.id) {
      toast.error('Oops! Something went wrong. Please try again.')
      return
    }
    setIsSubmitting(true)
    return requestPasswordResetEmail({
      id: user.id,
      isAdmin: !!isPromote
    })
      .then(res => {
        if (res?.data?.updateUserRole) {
          onSuccess?.()
        }
      })
      .finally(() => {
        setIsSubmitting(false)
      })
  }

  const title = isPromote ? 'Promote to Admin' : 'Demote to member'
  const userEmail = <span className="font-bold">&apos;{user?.email}&apos;</span>;
  const description = isPromote ? <>
    Are you sure you want to grant admin privileges to {userEmail}
  </> : <>
    Are you sure you want to demote {userEmail} to a regular member?
  </>;


  return (
    <AlertDialog open={open} onOpenChange={onOpenChange}>
      <AlertDialogContent>
        <AlertDialogHeader className="gap-3">
          <AlertDialogTitle>{title}</AlertDialogTitle>
          <AlertDialogDescription>
            {description}
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>Cancel</AlertDialogCancel>
          <AlertDialogAction
            className={buttonVariants()}
            onClick={onSubmit}
            disabled={isSubmitting}
          >
            {isSubmitting && (
              <IconSpinner className="mr-2 h-4 w-4 animate-spin" />
            )}
            Confirm
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  )
}
