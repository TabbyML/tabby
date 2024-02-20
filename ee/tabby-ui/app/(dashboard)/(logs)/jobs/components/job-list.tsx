'use client'

import React from 'react'
import { useQuery } from 'urql'

import { DEFAULT_PAGE_SIZE } from '@/lib/constants'
import { ListJobRunsQueryVariables } from '@/lib/gql/generates/graphql'
import { useIsQueryInitialized } from '@/lib/tabby/gql'
import { listJobRuns } from '@/lib/tabby/query'
import { Button } from '@/components/ui/button'
import { IconSpinner } from '@/components/ui/icons'
import { ListSkeleton } from '@/components/skeleton'

import { JobsTable } from './jobs-table'

const PAGE_SIZE = DEFAULT_PAGE_SIZE
export function JobRuns() {
  const [variables, setVariables] = React.useState<ListJobRunsQueryVariables>({
    last: PAGE_SIZE
  })
  const [{ data, error, fetching, stale }] = useQuery({
    query: listJobRuns,
    variables
  })

  const [initialized] = useIsQueryInitialized({ data, error, stale })

  const edges = data?.jobRuns?.edges
  const pageInfo = data?.jobRuns?.pageInfo
  const hasNextPage = pageInfo?.hasPreviousPage

  const fetchNextPage = () => {
    setVariables({ last: PAGE_SIZE, before: pageInfo?.startCursor })
  }

  const displayJobs = React.useMemo(() => {
    return edges?.slice().reverse()
  }, [edges])

  return (
    <>
      {initialized ? (
        <>
          <JobsTable jobs={displayJobs} />
          {hasNextPage && (
            <div className="mt-8 text-center">
              <Button disabled={fetching} onClick={fetchNextPage}>
                {fetching && (
                  <IconSpinner className="mr-2 h-4 w-4 animate-spin" />
                )}
                load more
              </Button>
            </div>
          )}
        </>
      ) : (
        <ListSkeleton />
      )}
    </>
  )
}
