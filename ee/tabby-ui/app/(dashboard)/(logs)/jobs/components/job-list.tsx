'use client'

import React from 'react'
import { useQuery } from 'urql'
import Link from 'next/link'
import humanizerDuration from 'humanize-duration'
import { isNil } from 'lodash-es'
import moment from 'moment'

import { cn } from '@/lib/utils'
import { ListJobRunsQueryVariables } from '@/lib/gql/generates/graphql'
import { useIsQueryInitialized } from '@/lib/tabby/gql'
import { listJobRuns } from '@/lib/tabby/query'

import {
  Table,
  TableBody,
  TableHead,
  TableHeader,
  TableRow,
  TableCell
} from '@/components/ui/table'
import { ListSkeleton } from '@/components/skeleton'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger
} from '@/components/ui/tooltip'
import { IconExternalLink } from '@/components/ui/icons'

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
      {initialized && (
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-[20%]">Name</TableHead>
              <TableHead>Recent Tasks</TableHead>
              <TableHead className="w-[20%]">Last Run</TableHead>
              <TableHead>Aggregate Runs</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {["Repository Index"].map((name, idx) => {
              // TODO: We only have one job type atm
              const jobs = displayJobs!

              const lastJob = jobs[0]
              const lastFinishedJob = jobs?.find(job => Boolean(job.node.finishedAt))
              const lastSuccessAt = lastFinishedJob
                ? moment(lastFinishedJob.node.finishedAt).format('YYYY-MM-DD HH:mm')
                : null
              return (
                <TableRow>
                  <TableCell className="font-bold">{name}</TableCell>
                  <TableCell>
                    <div className="flex">
                      {jobs?.map(job => {
                        const { createdAt, finishedAt } = job.node
                        const startAt =
                          createdAt && moment(createdAt).format('MMMM D, YYYY h:mm a')
                        const duration =
                          createdAt &&
                          finishedAt &&
                          humanizerDuration(
                            moment
                              .duration(moment(finishedAt).diff(createdAt))
                              .asMilliseconds()
                          )
                        return (
                          <TooltipProvider delayDuration={0} key={job.node.id}>
                            <Tooltip>
                              <TooltipTrigger asChild>
                                <Link
                                  href={`/jobs/detail?id=${job.node.id}`}
                                  className={cn(
                                    'mr-1 h-6 w-2 rounded-full hover:opacity-70',
                                    {
                                      'bg-blue-500': isNil(job.node.exitCode),
                                      'bg-green-500': job.node.exitCode === 0,
                                      'bg-red-500': job.node.exitCode === 1
                                    }
                                  )}
                                />
                              </TooltipTrigger>
                              <TooltipContent>
                                {startAt && <p>{startAt}</p>}
                                {duration && <p>Duration: {duration}</p>}
                              </TooltipContent>
                            </Tooltip>
                          </TooltipProvider>
                        )
                      })}
                    </div>
                  </TableCell>
                  <TableCell>
                    <Link
                      href={`/jobs/detail?id=${lastJob.node.id}`}
                      className="flex items-center hover:underline">
                      <p>{lastSuccessAt}</p>
                      <IconExternalLink className="ml-1" />
                    </Link>
                  </TableCell>
                  {/* TODO: mock aggregate data for the moment */}
                  <TableCell>
                    <div className="flex items-center gap-3">
                      <div className="flex h-8 w-8 items-center justify-center rounded-full border-2 border-green-500 text-xs text-green-500">
                        210
                      </div>
                      <div className="flex h-8 w-8 items-center justify-center rounded-full border-2 border-blue-500 text-xs text-blue-500">
                        1
                      </div>
                      <div className="flex h-8 w-8 items-center justify-center rounded-full border-2 border-muted text-xs text-muted">
                        
                      </div>
                    </div>
                  </TableCell>
                </TableRow>
              )
            })}
          </TableBody>
        </Table>
      )}
    </>
  )
}
