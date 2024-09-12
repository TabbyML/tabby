import { HTMLAttributes, useMemo, useState } from 'react'
import { CheckIcon } from '@radix-ui/react-icons'
import { useQuery } from 'urql'

import { graphql } from '@/lib/gql/generates'
import { UserGroupsQuery } from '@/lib/gql/generates/graphql'
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
import { IconChevronUpDown, IconEdit, IconSpinner } from '@/components/ui/icons'
import {
  Popover,
  PopoverContent,
  PopoverTrigger
} from '@/components/ui/popover'
import { Skeleton } from '@/components/ui/skeleton'
import LoadingWrapper from '@/components/loading-wrapper'
import { useMutation } from '@/lib/tabby/gql'

const listSourceIdAccessPolicies = graphql(/* GraphQL */ `
  query sourceIdAccessPolicies($sourceId: String!) {
    sourceIdAccessPolicies(sourceId: $sourceId) {
      sourceId
      read {
        id
        name
      }
    }
  }
`)

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

interface AccessPolicyViewProps extends HTMLAttributes<HTMLDivElement> {
  sourceId: string
  sourceName: string
  // onDelete: () => void
  // onGrant: () => void
  editable: boolean
  userGroups: UserGroupsQuery['userGroups'] | undefined
  fetchingUserGroups: boolean
}

export function AccessPolicyView({
  sourceId,
  sourceName,
  className,
  editable,
  userGroups,
  fetchingUserGroups,
  ...rest
}: AccessPolicyViewProps) {
  const [open, setOpen] = useState(false)
  const [{ data, fetching, error }, reexecuteQuery] = useQuery({
    query: listSourceIdAccessPolicies,
    variables: {
      sourceId
    }
  })

  const grantSourceIdReadAccess = useMutation(grantSourceIdReadAccessMutation)
  const revokeSourceIdReadAccess = useMutation(revokeSourceIdReadAccessMutation)

  const sourceIdAccessPolicies = data?.sourceIdAccessPolicies?.read
  const policiesLen = sourceIdAccessPolicies?.length || 0

  const [selectedIdSet, setSelectedIdSet] = useState<Set<string>>(new Set())

  // todo optimistic update
  // const selectedIdSet = useMemo(() => {
  //   if (!sourceIdAccessPolicies?.length) {
  //     return new Set()
  //   }

  //   return new Set(sourceIdAccessPolicies.map(policy => policy.id))
  // }, [sourceIdAccessPolicies])

  // FIXME demo code
  const handleSelectGroup = (id: string, grant: boolean) => {
    // FIXME grantSourceIdReadAccess & revokeSourceIdReadAccess
    setSelectedIdSet(prev => {
      const nextSet = new Set(prev)
      if (grant) {
        nextSet.add(id)
      } else {
        nextSet.delete(id)
      }
      return nextSet
    })
  }

  return (
    <LoadingWrapper
      loading={fetching}
      fallback={<Skeleton className={cn(className)} />}
    >
      <div className={cn('flex items-center gap-2', className)}>
        <span>{`${policiesLen} ${policiesLen <= 1 ? 'group' : 'groups'}`}</span>
        {editable && (
          <Popover open={open} onOpenChange={setOpen}>
            <PopoverTrigger asChild>
              <Button variant="ghost" role="combobox" size="icon">
                <IconEdit />
              </Button>
            </PopoverTrigger>
            <PopoverContent align="end" side="top">
              <Command className="transition-all">
                <CommandInput placeholder="Search groups..." />
                <CommandList className="max-h-[30vh]">
                  <CommandEmpty>
                    {fetchingUserGroups ? (
                      <div className="flex justify-center">
                        <IconSpinner className="h-6 w-6" />
                      </div>
                    ) : userGroups?.length ? (
                      'No matches results'
                    ) : (
                      'No groups found'
                    )}
                  </CommandEmpty>
                  <CommandGroup>
                    {userGroups?.map(group => {
                      const isSelected = selectedIdSet.has(group.id)
                      const memberLen = group.members.length
                      return (
                        <CommandItem
                          key={group.id}
                          onSelect={id => handleSelectGroup(id, !isSelected)}
                          value={group.id}
                        >
                          <div
                            className={cn(
                              'mr-2 flex h-4 w-4 items-center justify-center rounded-sm border border-primary cursor-pointer',
                              isSelected
                                ? 'bg-primary text-primary-foreground'
                                : 'opacity-50 [&_svg]:invisible'
                            )}
                          >
                            <CheckIcon className={cn('h-4 w-4')} />
                          </div>
                          <span>
                            {group.name}
                            <span className="text-muted-foreground ml-1">{`(${memberLen} member${memberLen > 1 ? 's' : ''
                              })`}</span>
                          </span>
                        </CommandItem>
                      )
                    })}
                  </CommandGroup>
                </CommandList>
              </Command>
            </PopoverContent>
          </Popover>
        )}
      </div>
    </LoadingWrapper>
  )
}
