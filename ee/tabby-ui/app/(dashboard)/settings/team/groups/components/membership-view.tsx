import {
  HTMLAttributes,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState
} from 'react'
import { pick } from 'lodash-es'
import { toast } from 'sonner'

import { graphql } from '@/lib/gql/generates'
import {
  UpsertUserGroupMembershipInput,
  User
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
  IconChevronUpDown,
  IconPlus,
  IconSpinner,
  IconTrash
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
import { Table, TableBody, TableCell, TableRow } from '@/components/ui/table'
import { UserAvatar } from '@/components/user-avatar'

import { MemberShips, MemberShipUser } from './types'
import { UserGroupItemContext } from './user-group-item'
import { UserGroupContext } from './user-group-page'

const deleteUserGroupMembershipMutation = graphql(/* GraphQL */ `
  mutation DeleteUserGroupMembership($userGroupId: ID!, $userId: ID!) {
    deleteUserGroupMembership(userGroupId: $userGroupId, userId: $userId)
  }
`)

const upsertUserGroupMembershipMutation = graphql(/* GraphQL */ `
  mutation UpsertUserGroupMembership($input: UpsertUserGroupMembershipInput!) {
    upsertUserGroupMembership(input: $input)
  }
`)

interface MembershipViewProps extends HTMLAttributes<HTMLDivElement> {
  userGroupId: string
  userGroupName: string
  members: MemberShips
}

export function MembershipView({ className, members }: MembershipViewProps) {
  const { isServerAdmin, isGroupAdmin } = useContext(UserGroupItemContext)
  const [emptyItemVisible, setEmptyItemVisible] = useState(false)
  const editable = isServerAdmin || isGroupAdmin

  const handleAddMember = () => {
    if (!emptyItemVisible) {
      setEmptyItemVisible(true)
    }
  }

  return (
    <div className={cn('flex flex-col gap-2 border-b px-1 py-2', className)}>
      <div className="max-h-[286px] flex-1 overflow-auto">
        {members.length || emptyItemVisible ? (
          <Table className="table-fixed">
            <TableBody>
              {members.map(member => {
                return (
                  <MembershipItem
                    key={member.user.id}
                    member={member}
                    onRemoveEmptyItem={() => setEmptyItemVisible(false)}
                  />
                )
              })}
              {emptyItemVisible && (
                <MembershipItem
                  onRemoveEmptyItem={() => setEmptyItemVisible(false)}
                />
              )}
            </TableBody>
          </Table>
        ) : (
          <div className="p-3 pl-4 text-muted-foreground">No members</div>
        )}
      </div>
      {editable && (
        <div className="mb-2 ml-2 flex justify-start">
          <Button
            variant="outline"
            disabled={emptyItemVisible}
            onClick={handleAddMember}
          >
            <IconPlus className="mr-2" />
            Add Member
          </Button>
        </div>
      )}
    </div>
  )
}

interface MembershipItemProps {
  member?: MemberShips[number]
  onRemoveEmptyItem: () => void
}

function roleToSelectItemValue(role: string) {
  return role === '1'
}

function MembershipItem({ member, onRemoveEmptyItem }: MembershipItemProps) {
  const { isServerAdmin, isGroupAdmin, memberIds, userGroupId } =
    useContext(UserGroupItemContext)

  const trRef = useRef<HTMLTableRowElement>(null)
  const [role, setRole] = useState(
    !isServerAdmin ? '0' : memberIds.length ? '0' : '1'
  )

  // update role after call upsert endpoint
  useEffect(() => {
    if (!!member && member.isGroupAdmin !== roleToSelectItemValue(role)) {
      setRole(member.isGroupAdmin ? '1' : '0')
    }
  }, [member])

  useEffect(() => {
    if (!member) {
      trRef.current?.scrollIntoView({
        behavior: 'smooth',
        block: 'nearest',
        inline: 'nearest'
      })
    }
  }, [])

  const deleteUserGroupMembership = useMutation(
    deleteUserGroupMembershipMutation
  )
  const upsertUserGroupMembership = useMutation(
    upsertUserGroupMembershipMutation
  )

  const handleDeleteUserGroupMembership = () => {
    if (!member) {
      onRemoveEmptyItem()
      return
    }
    deleteUserGroupMembership({
      userGroupId,
      userId: member.user.id
    })
      .then(res => {
        if (!res?.data?.deleteUserGroupMembership) {
          const errorMessage =
            res?.error?.message ||
            `Failed to delete ${member.user.name || member.user.email}`
          toast.error(errorMessage)
          return
        }
      })
      .catch(error => {
        toast.error(
          error.message ||
            `Failed to delete ${member.user.name || member.user.email}`
        )
      })
  }

  const handleUpsertUserGroupMembership = (
    isInsert: boolean,
    input: UpsertUserGroupMembershipInput,
    user: MemberShipUser
  ) => {
    const prevRole = role

    return upsertUserGroupMembership({ input, extraParams: { user, isInsert } })
      .then(res => {
        if (!res?.data?.upsertUserGroupMembership) {
          const errorMessage =
            res?.error?.message || `Failed to update ${user.name || user.email}`
          toast.error(errorMessage)
          setRole(prevRole)
          return
        }
      })
      .catch(error => {
        toast.error(
          error.message || `Failed to update ${user.name || user.email}`
        )
        setRole(prevRole)
      })
  }

  const onRoleChange = (role: string) => {
    setRole(role)

    if (member) {
      handleUpsertUserGroupMembership(
        false,
        {
          userGroupId,
          userId: member.user.id,
          isGroupAdmin: role === '1'
        },
        member.user
      )
    }
  }

  const onSelectMember = (userId: string, user: User) => {
    handleUpsertUserGroupMembership(
      true,
      {
        userId,
        userGroupId,
        isGroupAdmin: role === '1'
      },
      pick(user, 'id', 'email', 'createdAt', 'name')
    )
    onRemoveEmptyItem()
  }

  const deletable =
    isServerAdmin || (isGroupAdmin && (!member || !member.isGroupAdmin))

  return (
    <TableRow className="border-0 !bg-background pl-1" ref={trRef}>
      <TableCell>
        <MemberSelect membership={member} onChange={onSelectMember} />
      </TableCell>
      <TableCell className="w-[30%]">
        <Select
          onValueChange={onRoleChange}
          value={role}
          disabled={!isServerAdmin}
        >
          <SelectTrigger className="h-10">
            <SelectValue placeholder="Select role" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="1">Group Admin</SelectItem>
            <SelectItem value="0">Group Member</SelectItem>
          </SelectContent>
        </Select>
      </TableCell>
      <TableCell className="w-[100px] text-right">
        {deletable && (
          <Button
            size="icon"
            variant="hover-destructive"
            onClick={handleDeleteUserGroupMembership}
          >
            <IconTrash />
          </Button>
        )}
      </TableCell>
    </TableRow>
  )
}

function MemberSelect({
  membership,
  onChange
}: {
  membership?: MemberShips[number]
  onChange: (userId: string, user: User) => void
}) {
  const userId = membership?.user.id
  const { fetchingAllUsers, allUsers } = useContext(UserGroupContext)
  const { memberIds } = useContext(UserGroupItemContext)
  const [open, setOpen] = useState(false)
  const commandListRef = useRef<HTMLDivElement>(null)
  const options = useMemo(() => {
    return allUsers.filter(user => {
      return !memberIds.includes(user.id)
    })
  }, [memberIds, allUsers])
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

  const selectMember = (id: string) => {
    setOpen(false)
    onChange(id, options.find(user => user.id === id) as User)
  }

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          role="combobox"
          className={cn('h-10 w-full justify-between font-normal', {
            'text-muted-foreground hover:text-muted-foreground': !userId,
            'cursor-auto shadow-none hover:bg-background': !!userId
          })}
          onClick={e => {
            if (userId) {
              e.preventDefault()
            }
          }}
        >
          {fetchingAllUsers ? (
            'Loading...'
          ) : userId ? (
            <UserInfoView user={membership.user} />
          ) : (
            'Select member'
          )}
          {!userId && <IconChevronUpDown />}
        </Button>
      </PopoverTrigger>
      <PopoverContent
        className="w-[var(--radix-popover-trigger-width)] p-0"
        align="start"
        side="bottom"
      >
        <Command className="transition-all">
          <CommandInput
            placeholder="Search member..."
            onValueChange={onSearchChange}
          />
          <CommandList className="max-h-[30vh]" ref={commandListRef}>
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
              {options.map(user => (
                <CommandItem
                  key={user.id}
                  onSelect={() => selectMember(user.id)}
                >
                  <UserInfoView user={user} />
                </CommandItem>
              ))}
            </CommandGroup>
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  )
}

function UserInfoView({ user }: { user: MemberShips[0]['user'] }) {
  const userName = user.name
  return (
    <div className="flex h-10 items-center gap-2">
      <UserAvatar user={user} className="h-7 w-7" />
      <span className="space-x-1">
        {user.name}
        <span
          className={cn('text-sm', {
            'text-muted-foreground': !!userName
          })}
        >
          {userName ? `(${user.email})` : user.email}
        </span>
      </span>
    </div>
  )
}
