'use client'

import * as React from 'react'
import { zodResolver } from '@hookform/resolvers/zod'
import { useForm } from 'react-hook-form'
import * as z from 'zod'

import { graphql } from '@/lib/gql/generates'
import { useMutation } from '@/lib/tabby/gql'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger
} from '@/components/ui/dialog'
import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage
} from '@/components/ui/form'
import { IconSpinner } from '@/components/ui/icons'
import { Input } from '@/components/ui/input'

const createUserGroupMutation = graphql(/* GraphQL */ `
  mutation createUserGroup($input: CreateUserGroupInput!) {
    createUserGroup(input: $input)
  }
`)

const formSchema = z.object({
  name: z.string().trim()
})

export default function CreateUserGroupDialog({
  onSubmit,
  children
}: {
  onSubmit: (userId: string) => void
  children: React.ReactNode
}) {
  const [open, setOpen] = React.useState(false)
  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema)
  })

  const { isSubmitting } = form.formState

  const onOpenChange = (open: boolean) => {
    if (isSubmitting) {
      return
    }

    // reset form
    if (!open) {
      setTimeout(() => {
        form.reset()
      }, 500)
    }
    setOpen(open)
  }

  const createUserGroup = useMutation(createUserGroupMutation, {
    form
  })

  const handleSubmit = async (values: z.infer<typeof formSchema>) => {
    return createUserGroup({
      input: {
        name: values.name
      }
    })
      .then(res => {
        if (!res?.data?.createUserGroup) {
          return
        }

        onSubmit?.(res.data.createUserGroup)
        onOpenChange(false)
      })
      .catch(() => {})
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader className="gap-3">
          <DialogTitle>Create Group</DialogTitle>
        </DialogHeader>
        <Form {...form}>
          <div className="grid gap-2">
            <form
              className="grid gap-6"
              onSubmit={form.handleSubmit(handleSubmit)}
            >
              <div className="grid gap-2">
                <FormField
                  control={form.control}
                  name="name"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel required>Name</FormLabel>
                      <FormDescription>
                        Group name need to be unique, and it cannot be changed
                        after creation.
                      </FormDescription>
                      <FormControl>
                        <Input
                          placeholder="e.g backend-dev"
                          autoCapitalize="none"
                          autoCorrect="off"
                          autoComplete="off"
                          {...field}
                        />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                {/* root error message */}
                <FormMessage />
              </div>
              <div className="flex justify-end gap-4">
                <Button
                  type="button"
                  variant="ghost"
                  disabled={isSubmitting}
                  onClick={() => onOpenChange(false)}
                >
                  Cancel
                </Button>
                <Button type="submit" disabled={isSubmitting}>
                  {isSubmitting && <IconSpinner className="mr-2" />}
                  Create
                </Button>
              </div>
            </form>
          </div>
        </Form>
      </DialogContent>
      <DialogTrigger asChild>{children}</DialogTrigger>
    </Dialog>
  )
}
