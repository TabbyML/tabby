'use client'

import React from 'react'
import { useQuery } from 'urql'

import { ListJobRunsQueryVariables } from '@/lib/gql/generates/graphql'
import { useIsQueryInitialized } from '@/lib/tabby/gql'
import { listJobRuns } from '@/lib/tabby/query'
import { ListSkeleton } from '@/components/skeleton'

import {
  Table,
  TableBody,
  TableHead,
  TableHeader,
  TableRow
} from '@/components/ui/table'
import JobListRow from './job-list-row'

const PAGE_SIZE = 15

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
  
  const displayJobs = React.useMemo(() => {
    return edges?.slice().reverse()
  }, [edges])

  return (
    <>
      {!initialized && <ListSkeleton />}
      {initialized &&
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-[25%]">Name</TableHead>
              <TableHead className="w-[25%]">State</TableHead>
              <TableHead>Recent Tasks</TableHead>
              <TableHead className="w-[20%]">Last Success At</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            <JobListRow name="scheduler" jobs={displayJobs} />
          </TableBody>
        </Table>
      }
    </>
  )
}
