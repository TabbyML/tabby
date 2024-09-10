import { HTMLAttributes } from 'react'
import { useQuery } from 'urql'

import { graphql } from '@/lib/gql/generates'
import { cn } from '@/lib/utils'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { IconEdit, IconTrash, IconUser } from '@/components/ui/icons'

interface MembershipViewProps extends HTMLAttributes<HTMLDivElement> {
  userGroupId: string
}

const userGroupMembershipsQuery = graphql(/* GraphQL */ `
  query UserGroupMemberships($userGroupId: ID!) {
    userGroupMemberships(userGroupId: $userGroupId) {
      id
      userGroupId
      userId
      isGroupAdmin
      createdAt
      updatedAt
    }
  }
`)

export function MembershipView({
  userGroupId,
  className
}: MembershipViewProps) {
  const [{ data, fetching, error }] = useQuery({
    query: userGroupMembershipsQuery,
    variables: {
      userGroupId
    }
  })

  const memberships = data?.userGroupMemberships

  return (
    <div className={cn('border-b', className)}>
      {memberships?.length ? (
        memberships.map(item => {
          return (
            <div
              key={item.id}
              className="pl-10 pr-3 flex items-center gap-2 py-2 hover:bg-muted/50 border-b last:border-0"
            >
              <IconUser className="shrink-0" />
              <div className="flex-1">
                <div className="flex items-center gap-2 text-sm">
                  User Name
                  {item.isGroupAdmin ? (
                    <Badge>Group Admin</Badge>
                  ) : (
                    <Badge variant="secondary">Group Member</Badge>
                  )}
                </div>
                <div className="text-muted-foreground text-sm">
                  test@example.com
                </div>
              </div>
              <div className="flex items-center gap-2">
                <Button className="shrink-0" variant="ghost" size="icon">
                  <IconEdit />
                </Button>
                <Button
                  className="shrink-0"
                  variant="hover-destructive"
                  size="icon"
                >
                  <IconTrash />
                </Button>
              </div>
            </div>
          )
        })
      ) : (
        // todo
        <div className="p-3 text-center">No members</div>
      )}
    </div>
  )
}
