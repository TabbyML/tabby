'use client'

import { useMemo } from 'react'
import Link from 'next/link'
import { useQuery } from 'urql'
import moment from 'moment'
import humanizerDuration from 'humanize-duration'
import { isNil } from 'lodash-es'

import { cn } from '@/lib/utils'
import { listJobRuns, queryJobRunStats } from '@/lib/tabby/query'

import { IconExternalLink } from '@/components/ui/icons'
import { TableRow, TableCell } from '@/components/ui/table'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger
} from '@/components/ui/tooltip'
import { ListRowSkeleton } from '@/components/skeleton'
import LoadingWrapper from '@/components/loading-wrapper'

function JobAggregateState ({
  count,
  activeClass,
  tooltip,
}: {
  count?: number;
  activeClass: string;
  tooltip: string;
}) {
  return (
    <TooltipProvider delayDuration={0}>
      <Tooltip>
        <TooltipTrigger>
          <div className={cn("flex h-8 w-8 cursor-default items-center justify-center rounded-full border-2", {
            [activeClass]: count,
            "border-muted text-muted": !count
          })}>
            {count || ""}
          </div>
        </TooltipTrigger>
        <TooltipContent>
          <p>{tooltip}</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  )
}

function JobRunState ({
  name
}: {
  name: string
}) {
  const [{ data, fetching }] = useQuery({
    query: queryJobRunStats,
    variables: {
      jobs: [name]
    }
  })
  return (
    <LoadingWrapper
      loading={fetching}
      fallback={<ListRowSkeleton className="w-1/3" />}>
      <div className="flex items-center gap-3">
        <JobAggregateState
          count={data?.jobRunStats.success}
          activeClass="border-green-500 text-xs text-green-500"
          tooltip="Success" />
        <JobAggregateState
          count={data?.jobRunStats.pending}
          activeClass="border-blue-500 text-xs text-blue-500"
          tooltip="Pending" />
        <JobAggregateState
          count={data?.jobRunStats.failed}
          activeClass="border-red-500 text-xs text-red-500"
          tooltip="Failed" />
      </div>
    </LoadingWrapper>
  )
}

export default function JobRow ({
  name
}: {
  name: string
}) {
  const RECENT_DISPLAYED_SIZE = 10

  const [{ data, fetching }] = useQuery({
    query: listJobRuns,
    variables: {
      last: RECENT_DISPLAYED_SIZE,
      jobs: [name]
    }
  })

  const edges = data?.jobRuns?.edges
  const displayJobs = useMemo(() => {
    return edges?.slice().reverse()
  }, [edges])
  const lastJob = displayJobs?.[0]
  const lastFinishedJob = displayJobs?.find(job => Boolean(job.node.finishedAt))
  const lastSuccessAt = lastFinishedJob
    ? moment(lastFinishedJob.node.finishedAt).format('YYYY-MM-DD HH:mm')
    : null
  return (
    <LoadingWrapper
      loading={fetching}
      fallback={
        <TableRow>
          <TableCell colSpan={4}>
            <ListRowSkeleton />
          </TableCell>
        </TableRow>
    }>
      <TableRow className="h-16">
        <TableCell className="font-bold">{name}</TableCell>
        <TableCell>
          <div className="flex">
            {displayJobs?.map(job => {
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
            href={`/jobs/detail?id=${lastJob?.node.id}`}
            className="flex items-center hover:underline">
            <p>{lastSuccessAt}</p>
            <IconExternalLink className="ml-1" />
          </Link>
        </TableCell>
        <TableCell>
          <JobRunState name={name} />
        </TableCell>
      </TableRow>
    </LoadingWrapper>
  )
}
