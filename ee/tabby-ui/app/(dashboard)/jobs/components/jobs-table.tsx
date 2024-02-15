'use client'

import React from 'react'
import { useQuery } from 'urql'

import { DEFAULT_PAGE_SIZE } from '@/lib/constants'
import { useIsQueryInitialized } from '@/lib/tabby/gql'
import { listJobRuns } from '@/lib/tabby/query'
import { Button } from '@/components/ui/button'
import { IconSpinner } from '@/components/ui/icons'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow
} from '@/components/ui/table'
import { ListSkeleton } from '@/components/skeleton'
import { Badge } from '@/components/ui/badge'
import { ListJobRunsQueryVariables } from '@/lib/gql/generates/graphql'
import moment from 'moment'

const PAGE_SIZE = DEFAULT_PAGE_SIZE
export function JobRunsTable() {
  const [variables, setVariables] = React.useState<ListJobRunsQueryVariables>({ last: PAGE_SIZE })
  const [{ data, error, fetching }] = useQuery({
    query: listJobRuns,
    variables
  })
  const [initialized] = useIsQueryInitialized({ data, error })

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
    <div>
      {initialized ? (
        <>
          <Table>
            <TableHeader className='sticky top-0'>
              <TableRow>
                <TableHead className="w-[25%]">Time</TableHead>
                <TableHead className="w-[10%]">Job type</TableHead>
                <TableHead>Detail</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {!displayJobs?.length ? (
                <TableRow>
                  <TableCell colSpan={3} className="h-[100px] text-center">
                    No Data
                  </TableCell>
                </TableRow>
              ) : (
                <>
                  {displayJobs?.map(x => {
                    return (
                      <TableRow key={x.node.id}>
                        <TableCell>{moment(x.node.createdAt).format('YYYY-MM-DD HH:mm:ss Z')}</TableCell>
                        <TableCell><Badge variant='secondary'>{x.node.job}</Badge></TableCell>
                        <TableCell className="flex justify-end">
                          <>detail</>
                        </TableCell>
                      </TableRow>
                    )
                  })}
                </>
              )}
            </TableBody>
          </Table>
          {hasNextPage && (
            <div className='text-center mt-8'>
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
    </div>
  )
}
