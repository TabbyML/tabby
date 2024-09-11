'use client'

import * as React from 'react'
import { zodResolver } from '@hookform/resolvers/zod'
import { useForm } from 'react-hook-form'
import * as z from 'zod'

import { graphql } from '@/lib/gql/generates'
import { UserGroupMembership } from '@/lib/gql/generates/graphql'
import { useMutation } from '@/lib/tabby/gql'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList
} from '@/components/ui/command'
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage
} from '@/components/ui/form'
import {
  IconCheck,
  IconChevronUpDown,
  IconSpinner
} from '@/components/ui/icons'
import {
  Popover,
  PopoverContent,
  PopoverTrigger
} from '@/components/ui/popover'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from '@/components/ui/select'

import { UserGroupContext } from './user-group-page'

const upsertUserGroupMembershipMutation = graphql(/* GraphQL */ `
  mutation UpsertUserGroupMembership($input: UpsertUserGroupMembershipInput!) {
    upsertUserGroupMembership(input: $input)
  }
`)

const formSchema = z.object({
  userId: z.string(),
  isGroupAdmin: z.string()
})

type FormValues = z.infer<typeof formSchema>

interface UpsertMembershipFormProps {
  isNew: boolean
  userGroupId: string
  onSuccess?: () => void
  onCancel: () => void
  initialValues?: UserGroupMembership
  existingMemberIds?: string[]
}

export default function UpsertMembershipForm({
  isNew,
  userGroupId,
  onSuccess,
  onCancel,
  existingMemberIds,
  initialValues
}: UpsertMembershipFormProps) {
  const { allUsers, fetchingAllUsers } = React.useContext(UserGroupContext)
  const [open, setOpen] = React.useState(false)
  const defaultValues: Partial<FormValues> | undefined = initialValues
    ? {
        isGroupAdmin: initialValues.isGroupAdmin ? '1' : '0',
        userId: initialValues.user.id
      }
    : undefined
  const form = useForm<FormValues>({
    resolver: zodResolver(formSchema),
    defaultValues
  })
  const selectedUserId = form.watch('userId')
  const { isSubmitting } = form.formState
  const commandListRef = React.useRef<HTMLDivElement>(null)
  const selectedUser = selectedUserId
    ? allUsers.find(o => o.id === selectedUserId)
    : undefined
  const upsertUserGroupMembership = useMutation(
    upsertUserGroupMembershipMutation,
    {
      form
    }
  )

  const onSubmit = (values: FormValues) => {
    const { isGroupAdmin, ...rest } = values
    const _isGroupAdmin = isGroupAdmin === '1'
    return upsertUserGroupMembership({
      input: {
        userGroupId,
        isGroupAdmin: _isGroupAdmin,
        ...rest
      }
    }).then(res => {
      if (res?.data?.upsertUserGroupMembership) {
        form.reset()
        onSuccess?.()
      }
    })
  }

  const scrollCommandListToTop = () => {
    requestAnimationFrame(() => {
      if (commandListRef.current) {
        commandListRef.current.scrollTop = 0
      }
    })
  }

  const onSearchChange = () => {
    scrollCommandListToTop()
  }

  return (
    <Form {...form}>
      <div className="grid gap-2">
        <form className="grid gap-6" onSubmit={form.handleSubmit(onSubmit)}>
          <FormField
            control={form.control}
            name="userId"
            render={({ field }) => (
              <FormItem className="flex flex-col gap-2">
                <FormLabel required>User</FormLabel>
                <Popover open={open} onOpenChange={setOpen}>
                  <PopoverTrigger asChild>
                    <FormControl>
                      <Button
                        variant="outline"
                        role="combobox"
                        className={cn(
                          'justify-between font-normal',
                          !field.value && 'text-muted-foreground'
                        )}
                        disabled={!isNew}
                      >
                        {fetchingAllUsers
                          ? 'Loading...'
                          : field.value
                          ? selectedUser?.name || selectedUser?.email
                          : 'Select member'}
                        <IconChevronUpDown />
                      </Button>
                    </FormControl>
                  </PopoverTrigger>
                  <PopoverContent
                    className="w-[var(--radix-popover-trigger-width)] p-0"
                    align="start"
                    side="bottom"
                    sideOffset={-4}
                  >
                    <Command className="transition-all">
                      <CommandInput
                        placeholder="Search repository..."
                        onValueChange={onSearchChange}
                      />
                      <CommandList
                        className="max-h-[30vh]"
                        ref={commandListRef}
                      >
                        <CommandEmpty>
                          {fetchingAllUsers ? (
                            <div className="flex justify-center">
                              <IconSpinner className="h-6 w-6" />
                            </div>
                          ) : (
                            'No user found'
                          )}
                        </CommandEmpty>
                        <CommandGroup>
                          {allUsers?.map(user => (
                            <CommandItem
                              key={user.id}
                              onSelect={() => {
                                form.setValue('userId', user.id)
                                setOpen(false)
                              }}
                              disabled={existingMemberIds?.includes(user.id)}
                            >
                              <IconCheck
                                className={cn(
                                  'mr-2',
                                  user.id === field.value
                                    ? 'opacity-100'
                                    : 'opacity-0'
                                )}
                              />
                              {user.name || user.email}
                            </CommandItem>
                          ))}
                        </CommandGroup>
                      </CommandList>
                    </Command>
                  </PopoverContent>
                </Popover>
                <FormMessage />
              </FormItem>
            )}
          />
          <FormField
            control={form.control}
            name="isGroupAdmin"
            render={({ field }) => (
              <FormItem>
                <FormLabel required>Role</FormLabel>
                <Select
                  onValueChange={field.onChange}
                  defaultValue={field.value}
                >
                  <FormControl>
                    <SelectTrigger>
                      <SelectValue placeholder="Select role" />
                    </SelectTrigger>
                  </FormControl>
                  <SelectContent>
                    <SelectItem value="1">Group Admin</SelectItem>
                    <SelectItem value="0">Group Member</SelectItem>
                  </SelectContent>
                </Select>
                <FormMessage />
              </FormItem>
            )}
          />
          <div className="flex justify-end gap-4">
            <Button
              type="button"
              variant="ghost"
              disabled={isSubmitting}
              onClick={onCancel}
            >
              Cancel
            </Button>
            <Button type="submit" disabled={isSubmitting}>
              {isNew ? 'Add' : 'Update'}
            </Button>
          </div>
        </form>
        <FormMessage className="text-center" />
      </div>
    </Form>
  )
}
