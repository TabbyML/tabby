'use client'

import React from 'react'
import { zodResolver } from '@hookform/resolvers/zod'
import { isEmpty } from 'lodash-es'
import { useForm } from 'react-hook-form'
import { toast } from 'sonner'
import * as z from 'zod'

import { graphql } from '@/lib/gql/generates/gql'
import { useMutation } from '@/lib/tabby/gql'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  // FormItem,
  FormLabel,
  FormMessage
} from '@/components/ui/form'
import { IconSpinner } from '@/components/ui/icons'
import { Label } from '@/components/ui/label'
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group'

const updateUserRoleMutation = graphql(/* GraphQL */ `
  mutation updateUserRole($id: ID!, $isAdmin: Boolean!) {
    updateUserRole(id: $id, isAdmin: $isAdmin)
  }
`)

const formSchema = z.object({
  isAdmin: z.string()
})

type FormValeus = z.infer<typeof formSchema>
interface UpdateUserRoleDialogProps {
  open?: boolean
  onOpenChange?(open: boolean): void
  user?: { id: string; email: string; isAdmin: boolean }
  onSuccess?: () => void
}

export const UpdateUserRoleDialog: React.FC<UpdateUserRoleDialogProps> = ({
  user,
  onSuccess,
  open,
  onOpenChange
}) => {
  const form = useForm<FormValeus>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      isAdmin: user?.isAdmin ? '1' : '0'
    }
  })

  const isDirty = !isEmpty(form.formState.dirtyFields)
  const { isSubmitting } = form.formState

  const requestPasswordResetEmail = useMutation(updateUserRoleMutation, {
    form
  })

  const onSubmit = (values: FormValeus) => {
    if (!user?.id) return
    return requestPasswordResetEmail({
      id: user.id,
      isAdmin: values?.isAdmin === '1'
    }).then(res => {
      if (res?.data?.updateUserRole) {
        toast.success('User role is updated.')
        onSuccess?.()
      }
    })
  }

  React.useEffect(() => {
    if (open) {
      form.reset({ isAdmin: user?.isAdmin ? '1' : '0' })
    }
  }, [open])

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader className="gap-3">
          <DialogTitle>Edit Role</DialogTitle>
          <DialogDescription>
            Update role of user {user?.email}
          </DialogDescription>
        </DialogHeader>
        <Form {...form}>
          <form className="grid gap-2" onSubmit={form.handleSubmit(onSubmit)}>
            <FormField
              control={form.control}
              name="isAdmin"
              render={({ field: { onChange, ...rest } }) => (
                <FormItem>
                  <FormLabel>Role</FormLabel>
                  <FormControl>
                    <RadioGroup
                      className="flex gap-6"
                      orientation="horizontal"
                      onValueChange={onChange}
                      {...rest}
                    >
                      <div className="flex items-center space-x-2">
                        <RadioGroupItem value={'1'} id="admin" />
                        <Label className="cursor-pointer" htmlFor="admin">
                          Admin
                        </Label>
                      </div>
                      <div className="flex items-center space-x-2">
                        <RadioGroupItem value={'0'} id="member" />
                        <Label className="cursor-pointer" htmlFor="member">
                          Member
                        </Label>
                      </div>
                    </RadioGroup>
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            <Button
              type="submit"
              className="mt-2"
              disabled={!isDirty || isSubmitting}
            >
              {isSubmitting && (
                <IconSpinner className="mr-2 h-4 w-4 animate-spin" />
              )}
              Update
            </Button>
          </form>
          <FormMessage className="text-center" />
        </Form>
      </DialogContent>
    </Dialog>
  )
}
