'use client'

import React from 'react'
import Link from 'next/link'
import { useParams } from 'next/navigation'
import { useQuery } from 'urql'

import { DEFAULT_PAGE_SIZE } from '@/lib/constants'
import {
  IntegrationStatus,
  ListIntegrationsQuery
} from '@/lib/gql/generates/graphql'
import { listIntegrations } from '@/lib/tabby/query'
import { buttonVariants } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { LoadMoreIndicator } from '@/components/load-more-indicator'
import LoadingWrapper from '@/components/loading-wrapper'

import { useIntegrationKind } from '../hooks/use-repository-kind'

const PAGE_SIZE = DEFAULT_PAGE_SIZE

export default function RepositoryProvidersPage() {
  const kind = useIntegrationKind()
  const params = useParams()
  const [lastCursor, setLastCursor] = React.useState<string | undefined>(
    undefined
  )
  const [{ data, fetching }] = useQuery({
    query: listIntegrations,
    variables: { kind, last: PAGE_SIZE, before: lastCursor }
  })

  const edges = React.useMemo(() => {
    return data?.integrations?.edges?.slice().reverse()
  }, [data?.integrations?.edges])
  const pageInfo = data?.integrations?.pageInfo

  const loadMore = () => {
    if (pageInfo?.startCursor) {
      setLastCursor(pageInfo.startCursor)
    }
  }

  return (
    <LoadingWrapper loading={fetching} fallback={<FetchingSkeletion />}>
      {edges?.length ? (
        <>
          <CreateRepositoryProvider />
          <div className="space-y-8">
            {edges?.map(item => {
              return (
                <Card key={item.node.id}>
                  <CardHeader className="border-b px-6 py-4">
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-xl">
                        <div className="flex items-center gap-2">
                          {item.node.displayName}
                        </div>
                      </CardTitle>
                      <Link
                        href={`${params.kind}/detail?id=${item.node.id}`}
                        className={buttonVariants({ variant: 'secondary' })}
                      >
                        View
                      </Link>
                    </div>
                  </CardHeader>
                  <CardContent className="p-0 text-sm">
                    <div className="flex px-6 py-4">
                      <span className="w-[30%] shrink-0 text-muted-foreground">
                        Status
                      </span>
                      <span>{toStatusMessage(item.node)}</span>
                    </div>
                  </CardContent>
                </Card>
              )
            })}
            {!!pageInfo?.hasPreviousPage && (
              <LoadMoreIndicator
                itemCount={edges.length}
                onLoad={loadMore}
                isFetching={fetching}
              >
                <FetchingSkeletion />
              </LoadMoreIndicator>
            )}
          </div>
        </>
      ) : (
        <ProvidersPlaceholder />
      )}
    </LoadingWrapper>
  )
}

function CreateRepositoryProvider() {
  const params = useParams()
  return (
    <div className="my-4 flex justify-end">
      <Link href={`./${params.kind}/new`} className={buttonVariants()}>
        Create
      </Link>
    </div>
  )
}

function toStatusMessage(
  node: ListIntegrationsQuery['integrations']['edges'][0]['node']
) {
  switch (node.status) {
    case IntegrationStatus.Ready:
      return 'Ready'
    case IntegrationStatus.Failed:
      return (
        node.message ||
        'Processing error. Please check if the access token is still valid'
      )
    case IntegrationStatus.Pending:
      return 'Awaiting the next data synchronization'
  }
}

function ProvidersPlaceholder() {
  const params = useParams()
  return (
    <div className="flex flex-col items-center gap-4 rounded-lg border-4 border-dashed py-8">
      <div>No Data</div>
      <div className="flex justify-center">
        <Link
          href={`./${params.kind}/new`}
          className={buttonVariants({ variant: 'default' })}
        >
          Create
        </Link>
      </div>
    </div>
  )
}

function CardSkeleton() {
  return (
    <Card className="w-full bg-transparent">
      <CardHeader className="border-b px-6 py-4">
        <CardTitle>
          <Skeleton className="w-[20%]" />
        </CardTitle>
      </CardHeader>
      <CardContent className="px-6 py-4">
        <Skeleton className="w-[80%]" />
      </CardContent>
    </Card>
  )
}

function FetchingSkeletion() {
  return (
    <div className="space-y-8">
      <CardSkeleton />
      <CardSkeleton />
    </div>
  )
}
