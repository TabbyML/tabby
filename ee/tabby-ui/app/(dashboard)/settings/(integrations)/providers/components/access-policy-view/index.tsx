import { HTMLAttributes, useState } from 'react'
import { useQuery } from 'urql'

import { graphql } from '@/lib/gql/generates'
import { UserGroupsQuery } from '@/lib/gql/generates/graphql'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger
} from '@/components/ui/dialog'
import { IconEdit } from '@/components/ui/icons'
import { Skeleton } from '@/components/ui/skeleton'
import LoadingWrapper from '@/components/loading-wrapper'

import { ReadAccessPoliciesTable } from './access-policies-table'

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

  const sourceIdAccessPolicies = data?.sourceIdAccessPolicies?.read
  const policiesLen = sourceIdAccessPolicies?.length || 0

  return (
    <LoadingWrapper
      loading={fetching}
      fallback={<Skeleton className={cn(className)} />}
    >
      <div className={cn('flex items-center gap-2', className)}>
        {/* todo hovercard */}
        <span>{`${policiesLen} ${policiesLen <= 1 ? 'group' : 'groups'}`}</span>
        {editable && (
          <Dialog open={open} onOpenChange={setOpen}>
            <DialogContent className="w-[60vw] max-w-none">
              <DialogHeader className="gap-3">
                <DialogTitle>Read Access: {`'${sourceName}'`}</DialogTitle>
              </DialogHeader>
              <ReadAccessPoliciesTable
                onUpdate={() => {}}
                userGroups={userGroups}
                fetchingUserGroups={fetchingUserGroups}
                sourceId={sourceId}
                readAccessPolicies={sourceIdAccessPolicies}
              />
            </DialogContent>
            <DialogTrigger asChild>
              <Button size="icon" variant="ghost">
                <IconEdit />
              </Button>
            </DialogTrigger>
          </Dialog>
        )}
      </div>
    </LoadingWrapper>
  )
}
