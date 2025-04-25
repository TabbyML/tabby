'use client'

import { useMemo } from 'react'
import Link from 'next/link'
import humanizerDuration from 'humanize-duration'
import { isNil } from 'lodash-es'
import moment from 'moment'
import useSWR from 'swr'
import { useQuery } from 'urql'

import { listJobRuns, queryJobRunStats } from '@/lib/tabby/query'
import { cn } from '@/lib/utils'
import { IconSpinner } from '@/components/ui/icons'
import { TableCell, TableRow } from '@/components/ui/table'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger
} from '@/components/ui/tooltip'
import LoadingWrapper from '@/components/loading-wrapper'
import { ListRowSkeleton } from '@/components/skeleton'

import { getJobDisplayName } from '../utils'

function JobAggregateState({
  count,
  activeClass,
  tooltip
}: {
  count?: number
  activeClass: string
  tooltip: string
}) {
  return (
    <TooltipProvider delayDuration={0}>
      <Tooltip>
        <TooltipTrigger>
          <div
            className={cn(
              'flex h-8 w-8 cursor-default items-center justify-center rounded-full',
              activeClass
            )}
          >
            {count || 0}
          </div>
        </TooltipTrigger>
        <TooltipContent>
          <p>{tooltip}</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  )
}

function JobRunState({ name }: { name: string }) {
  const [{ data, fetching }] = useQuery({
    query: queryJobRunStats,
    variables: {
      jobs: [name]
    }
  })
  return (
    <LoadingWrapper
      loading={fetching}
      fallback={<ListRowSkeleton className="w-1/3" />}
    >
      <div className="flex items-center gap-3">
        <JobAggregateState
          count={data?.jobRunStats.success}
          activeClass="bg-green-600 text-xs text-white"
          tooltip="Success"
        />
        <JobAggregateState
          count={data?.jobRunStats.pending}
          activeClass="bg-blue-600 text-xs text-white"
          tooltip="Pending"
        />
        <JobAggregateState
          count={data?.jobRunStats.failed}
          activeClass="bg-red-600 text-xs text-white"
          tooltip="Failed"
        />
      </div>
    </LoadingWrapper>
  )
}

export default function JobRow({ name }: { name: string }) {
  const RECENT_DISPLAYED_SIZE = 10

  const [{ data, fetching }, reexecuteQuery] = useQuery({
    query: listJobRuns,
    variables: {
      last: RECENT_DISPLAYED_SIZE,
      jobs: [name]
    }
  })

  useSWR('refresh_jobs', () => reexecuteQuery(), {
    revalidateOnFocus: true,
    revalidateOnReconnect: true,
    revalidateOnMount: false,
    refreshInterval: 10 * 1000
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
      }
    >
      <TableRow className="h-16">
        <TableCell className="font-bold">{getJobDisplayName(name)}</TableCell>
        <TableCell>
          <div className="grid grid-cols-5 flex-wrap gap-1.5  xl:flex">
            {displayJobs?.map(job => {
              const { createdAt, finishedAt, startedAt } = job.node
              const isJobRunning = !finishedAt && !!startedAt
              const createAt =
                createdAt && moment(createdAt).format('YYYY-MM-DD HH:mm')
              const duration: string | null =
                (startedAt &&
                  finishedAt &&
                  humanizerDuration.humanizer({
                    language: 'shortEn',
                    languages: {
                      shortEn: {
                        d: () => 'd',
                        h: () => 'h',
                        m: () => 'm',
                        s: () => 's'
                      }
                    }
                  })(
                    moment
                      .duration(moment(finishedAt).diff(startedAt))
                      .asMilliseconds(),
                    {
                      units: ['d', 'h', 'm', 's'],
                      round: false,
                      largest: 1,
                      spacer: '',
                      language: 'shortEn'
                    }
                  )) ??
                null

              let displayedDuration = ''
              if (duration !== null) {
                const isSecond = duration.endsWith('s')
                if (isSecond) {
                  displayedDuration = duration
                } else {
                  const unit = duration.charAt(duration.length - 1)
                  const roundNumber = parseInt(duration) + 1
                  displayedDuration = roundNumber + unit
                }
              }

              return (
                <TooltipProvider delayDuration={0} key={job.node.id}>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Link
                        href={`/jobs/detail?id=${job.node.id}`}
                        className={cn(
                          'flex h-8 w-8 items-center justify-center rounded text-xs text-white hover:opacity-70',
                          {
                            'bg-blue-600': isNil(job.node.exitCode),
                            'bg-green-600': job.node.exitCode === 0,
                            'bg-red-600':
                              !isNil(job.node.exitCode) &&
                              job.node.exitCode !== 0
                          }
                        )}
                      >
                        {displayedDuration}
                        {isJobRunning && <IconSpinner />}
                      </Link>
                    </TooltipTrigger>
                    <TooltipContent>
                      {createAt && <p>{createAt}</p>}
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
            className="flex items-center underline"
          >
            <p>{lastSuccessAt}</p>
          </Link>
        </TableCell>
        <TableCell>
          <JobRunState name={name} />
        </TableCell>
      </TableRow>
    </LoadingWrapper>
  )
}
