import React from 'react'
import Link from 'next/link'
import moment from 'moment'

import { ListJobRunsQuery } from '@/lib/gql/generates/graphql'
import { Badge } from '@/components/ui/badge'
import { buttonVariants } from '@/components/ui/button'
import { IconCheck, IconClose, IconPieChart } from '@/components/ui/icons'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow
} from '@/components/ui/table'

type TJobRun = ListJobRunsQuery['jobRuns']['edges'][0]
interface JobsTableProps {
  jobs: TJobRun[] | undefined
  shouldRedirect?: boolean
}

export const JobsTable: React.FC<JobsTableProps> = ({
  jobs,
  shouldRedirect = true
}) => {
  return (
    <Table>
      <TableHeader className="sticky top-0">
        <TableRow>
          <TableHead className="w-[200px]">Start Time</TableHead>
          <TableHead className="w-[100px]">Duration</TableHead>
          <TableHead className="w-[100px]">Job Type</TableHead>
          <TableHead className="w-[100px] text-center">Status</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {!jobs?.length ? (
          <TableRow>
            <TableCell
              colSpan={shouldRedirect ? 4 : 3}
              className="h-[100px] text-center"
            >
              No Data
            </TableCell>
          </TableRow>
        ) : (
          <>
            {jobs?.map(x => {
              const duration = getJobDuration(x.node)
              return (
                <TableRow key={x.node.id}>
                  <TableCell>
                    {moment(x.node.createdAt).format('YYYY-MM-DD HH:mm:ss Z')}
                  </TableCell>
                  <TableCell>{duration ? `${duration}` : 'pending'}</TableCell>
                  <TableCell>
                    <Badge variant="secondary">{x.node.job}</Badge>
                  </TableCell>
                  <TableCell className="flex justify-center">
                    <JobStatusAction node={x} shouldRedirect={shouldRedirect}>
                      <JobStatusIcon node={x} />
                    </JobStatusAction>
                  </TableCell>
                </TableRow>
              )
            })}
          </>
        )}
      </TableBody>
    </Table>
  )
}

function getJobDuration({
  createdAt,
  finishedAt
}: {
  createdAt: string
  finishedAt?: string
}) {
  if (!finishedAt) return undefined

  let duration = moment.duration(moment(finishedAt).diff(createdAt))
  return formatDuration(duration)
}

function formatDuration(duration: moment.Duration) {
  const hours = duration.hours()
  const minutes = duration.minutes()
  const seconds = duration.seconds()

  let formattedDuration = ''

  if (hours > 0) {
    formattedDuration += `${hours}h`
  }

  if (minutes > 0) {
    if (formattedDuration.length > 0) {
      formattedDuration += ' '
    }

    formattedDuration += `${minutes}min`
  }

  if (seconds > 0) {
    if (formattedDuration.length > 0) {
      formattedDuration += ' '
    }

    formattedDuration += `${seconds}s`
  }

  return formattedDuration
}

function JobStatusIcon({ node }: { node: TJobRun }) {
  if (!node) return null
  const finishedAt = node?.node?.finishedAt
  const exitCode = node?.node?.exitCode

  // runing, success, error
  if (!finishedAt) {
    return <IconPieChart />
  }
  if (exitCode === 0) {
    return <IconCheck className="text-successful-foreground" />
  }

  return <IconClose className="text-destructive-foreground" />
}

function JobStatusAction({
  node,
  shouldRedirect,
  children
}: {
  shouldRedirect?: boolean
  node: TJobRun
  children: React.ReactNode
}) {
  if (shouldRedirect) {
    return (
      <Link
        className={buttonVariants({ variant: 'ghost' })}
        href={`/jobs/detail?id=${node.node.id}`}
      >
        {children}
      </Link>
    )
  }
  return children
}
