import { useRef, useState } from 'react'
import { zodResolver } from '@hookform/resolvers/zod'
import { useForm } from 'react-hook-form'
import * as z from 'zod'

import { graphql } from '@/lib/gql/generates'
import {
  SourceIdAccessPoliciesQuery,
  UserGroupsQuery
} from '@/lib/gql/generates/graphql'
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
  FormMessage
} from '@/components/ui/form'
import {
  IconCheck,
  IconChevronUpDown,
  IconSpinner,
  IconTrash
} from '@/components/ui/icons'
import {
  Popover,
  PopoverContent,
  PopoverTrigger
} from '@/components/ui/popover'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow
} from '@/components/ui/table'

const grantSourceIdReadAccessMutation = graphql(/* GraphQL */ `
  mutation grantSourceIdReadAccess($sourceId: String!, $userGroupId: ID!) {
    grantSourceIdReadAccess(sourceId: $sourceId, userGroupId: $userGroupId)
  }
`)

const revokeSourceIdReadAccessMutation = graphql(/* GraphQL */ `
  mutation revokeSourceIdReadAccess($sourceId: String!, $userGroupId: ID!) {
    revokeSourceIdReadAccess(sourceId: $sourceId, userGroupId: $userGroupId)
  }
`)

interface ReadAccessPoliciesTableProps {
  sourceId: string
  readAccessPolicies:
    | SourceIdAccessPoliciesQuery['sourceIdAccessPolicies']['read']
    | undefined
  onUpdate: () => void
  userGroups: UserGroupsQuery['userGroups'] | undefined
  fetchingUserGroups: boolean
}

const formSchema = z.object({
  userGroupId: z.string()
})

type GrantSourceIdReadAccessFormValues = z.infer<typeof formSchema>

export function ReadAccessPoliciesTable({
  sourceId,
  readAccessPolicies,
  onUpdate,
  userGroups,
  fetchingUserGroups
}: ReadAccessPoliciesTableProps) {
  const [open, setOpen] = useState(false)
  const form = useForm<GrantSourceIdReadAccessFormValues>({
    resolver: zodResolver(formSchema)
  })
  const commandListRef = useRef<HTMLDivElement>(null)
  const { isSubmitting } = form.formState

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

  const grantSourceIdReadAccess = useMutation(grantSourceIdReadAccessMutation, {
    form
  })
  const revokeSourceIdReadAccess = useMutation(revokeSourceIdReadAccessMutation)

  const onGrant = (values: GrantSourceIdReadAccessFormValues) => {
    return grantSourceIdReadAccess({
      sourceId,
      userGroupId: values.userGroupId
    })
  }

  const onRevoke = (userGroupId: string) => {
    return revokeSourceIdReadAccess({
      sourceId,
      userGroupId
    })
  }

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* FIXME: height */}
      <Table className="h-[300px]">
        <TableHeader>
          <TableRow>
            <TableHead>Granted User Groups</TableHead>
            <TableHead className="w-[80px]"></TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {readAccessPolicies?.length ? (
            readAccessPolicies?.map(policy => {
              return (
                <TableRow key={policy.id}>
                  <TableCell className="break-all lg:break-words">
                    {policy.name}
                  </TableCell>
                  <TableCell>
                    <Button
                      size="icon"
                      variant="hover-destructive"
                      onClick={() => {
                        onRevoke(policy.id)
                      }}
                    >
                      <IconTrash />
                    </Button>
                  </TableCell>
                </TableRow>
              )
            })
          ) : (
            <TableRow>
              <TableCell colSpan={2} className="text-center font-semibold">
                No Data
              </TableCell>
            </TableRow>
          )}
        </TableBody>
      </Table>
      <Form {...form}>
        <div className="grid gap-2 mt-8">
          <form
            className="flex items-center gap-4"
            onSubmit={form.handleSubmit(onGrant)}
          >
            <FormField
              control={form.control}
              name="userGroupId"
              render={({ field }) => (
                <FormItem className="flex flex-col">
                  <Popover open={open} onOpenChange={setOpen}>
                    <PopoverTrigger asChild>
                      <FormControl>
                        <Button
                          variant="outline"
                          role="combobox"
                          className={cn(
                            'justify-between font-normal w-[220px]',
                            !field.value && 'text-muted-foreground'
                          )}
                        >
                          {field.value
                            ? userGroups?.find(
                                group => group.id === field.value
                              )?.name
                            : 'Add user group'}
                          <IconChevronUpDown />
                        </Button>
                      </FormControl>
                    </PopoverTrigger>
                    <PopoverContent
                      className="w-[var(--radix-popover-trigger-width)] p-0"
                      align="start"
                      side="bottom"
                    >
                      <Command className="transition-all">
                        <CommandInput
                          placeholder="Search user group..."
                          onValueChange={onSearchChange}
                        />
                        <CommandList
                          className="max-h-[30vh]"
                          ref={commandListRef}
                        >
                          <CommandEmpty>
                            {fetchingUserGroups ? (
                              <div className="flex justify-center">
                                <IconSpinner className="h-6 w-6" />
                              </div>
                            ) : (
                              'No matches user group'
                            )}
                          </CommandEmpty>
                          <CommandGroup>
                            {userGroups?.map(group => (
                              <CommandItem
                                key={group.id}
                                onSelect={() => {
                                  form.setValue('userGroupId', group.id)
                                  setOpen(false)
                                }}
                              >
                                <IconCheck
                                  className={cn(
                                    'mr-2',
                                    group.id === field.value
                                      ? 'opacity-100'
                                      : 'opacity-0'
                                  )}
                                />
                                {group.name}
                              </CommandItem>
                            ))}
                          </CommandGroup>
                        </CommandList>
                      </Command>
                    </PopoverContent>
                  </Popover>
                  <FormMessage className="text-left" />
                </FormItem>
              )}
            />
            <div className="flex justify-end gap-4">
              <Button type="submit" disabled={isSubmitting}>
                Add
              </Button>
            </div>
          </form>
          <FormMessage className="text-left" />
        </div>
      </Form>
    </div>
  )
}
