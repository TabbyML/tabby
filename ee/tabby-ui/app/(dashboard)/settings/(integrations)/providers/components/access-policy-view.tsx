import { HTMLAttributes, useMemo, useState } from 'react'
import { CheckIcon } from '@radix-ui/react-icons'
import { toast } from 'sonner'
import { useQuery } from 'urql'

import { graphql } from '@/lib/gql/generates'
import { UserGroupsQuery } from '@/lib/gql/generates/graphql'
import { useMutation } from '@/lib/tabby/gql'
import { listSourceIdAccessPolicies } from '@/lib/tabby/query'
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
import { IconEdit, IconSpinner } from '@/components/ui/icons'
import {
  Popover,
  PopoverContent,
  PopoverTrigger
} from '@/components/ui/popover'
import { Skeleton } from '@/components/ui/skeleton'
import LoadingWrapper from '@/components/loading-wrapper'

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
  const [{ data, fetching }] = useQuery({
    query: listSourceIdAccessPolicies,
    variables: {
      sourceId
    }
  })

  const grantSourceIdReadAccess = useMutation(grantSourceIdReadAccessMutation)
  const revokeSourceIdReadAccess = useMutation(revokeSourceIdReadAccessMutation)

  const sourceIdAccessPolicies = data?.sourceIdAccessPolicies?.read
  const policiesLen = sourceIdAccessPolicies?.length || 0

  const selectedIdSet = useMemo(() => {
    if (!sourceIdAccessPolicies?.length) {
      return new Set()
    }

    return new Set(sourceIdAccessPolicies.map(policy => policy.id))
  }, [sourceIdAccessPolicies])

  const handleSelectGroup = (
    userGroupId: string,
    userGroupName: string,
    grant: boolean
  ) => {
    if (grant) {
      onGrantSourceIdReadAccess(userGroupId, userGroupName)
    } else {
      onRevokeSourceIdReadAccess(userGroupId, userGroupName)
    }
  }

  const onGrantSourceIdReadAccess = (
    userGroupId: string,
    userGroupName: string
  ) => {
    const defaultErrorMessage = `Failed to grant ${userGroupName}`
    return grantSourceIdReadAccess(
      {
        sourceId,
        userGroupId
      },
      {
        extraParams: {
          userGroupName
        }
      }
    )
      .then(res => {
        if (!res?.data?.grantSourceIdReadAccess) {
          const errorMessage = res?.error?.message || defaultErrorMessage
          toast.error(errorMessage)
          return
        }
      })
      .catch(error => {
        const errorMessage = error?.message || defaultErrorMessage
        toast.error(errorMessage)
      })
  }

  const onRevokeSourceIdReadAccess = (
    userGroupId: string,
    userGroupName: string
  ) => {
    return revokeSourceIdReadAccess(
      {
        sourceId,
        userGroupId
      },
      {
        extraParams: {
          userGroupName
        }
      }
    )
      .then(res => {
        if (!res?.data?.revokeSourceIdReadAccess) {
          const errorMessage =
            res?.error?.message || `Failed to revoke '${userGroupName}'`
          toast.error(errorMessage)
          return
        }
      })
      .catch(error => {
        const errorMessage =
          error?.message || `Failed to revoke '${userGroupName}'`
        toast.error(errorMessage)
      })
  }

  let accessText =
    policiesLen === 0
      ? 'Everyone'
      : `${policiesLen} ${policiesLen <= 1 ? 'group' : 'groups'}`

  return (
    <LoadingWrapper
      loading={fetching}
      fallback={<Skeleton className={cn(className)} />}
    >
      <div className={cn('flex items-center gap-0.5', className)}>
        <span className="w-[68px]">{accessText}</span>
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
                          value={group.name}
                          onSelect={() =>
                            handleSelectGroup(group.id, group.name, !isSelected)
                          }
                        >
                          <div
                            className={cn(
                              'mr-2 flex h-4 w-4 cursor-pointer items-center justify-center rounded-sm border border-primary',
                              isSelected
                                ? 'bg-primary text-primary-foreground'
                                : 'opacity-50 [&_svg]:invisible'
                            )}
                          >
                            <CheckIcon className={cn('h-4 w-4')} />
                          </div>
                          <span>
                            {group.name}
                            <span className="ml-1 text-muted-foreground">{`(${memberLen} member${
                              memberLen > 1 ? 's' : ''
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
