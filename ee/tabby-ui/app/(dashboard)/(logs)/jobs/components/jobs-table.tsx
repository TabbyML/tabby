import React from 'react'
import { useRouter } from 'next/navigation'
import moment from 'moment'

import { ListJobRunsQuery } from '@/lib/gql/generates/graphql'
import { cn } from '@/lib/utils'
import { Badge } from '@/components/ui/badge'
import { IconCheck, IconClose, IconSpinner } from '@/components/ui/icons'
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
  const router = useRouter()
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead className="w-[200px]">Start Time</TableHead>
          <TableHead className="w-[100px]">Duration</TableHead>
          <TableHead className="w-[100px]">Job</TableHead>
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
                <TableRow
                  key={x.node.id}
                  className={cn(shouldRedirect && 'cursor-pointer')}
                  onClick={e => {
                    if (shouldRedirect) {
                      router.push(`/jobs/detail?id=${x.node.id}`)
                    }
                  }}
                >
                  <TableCell>
                    {moment(x.node.createdAt).format('MMMM D, YYYY h:mm a')}
                  </TableCell>
                  <TableCell>{x.node?.finishedAt ? 'Running' : `${duration ?? '-'}`}</TableCell>
                  <TableCell>
                    <Badge variant="secondary">{x.node.job}</Badge>
                  </TableCell>
                  <TableCell>
                    <div className="flex items-center justify-center">
                      <JobStatusIcon node={x} />
                    </div>
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
    return <IconSpinner />
  }
  if (exitCode === 0) {
    return <IconCheck className="text-successful-foreground" />
  }

  return <IconClose className="text-destructive-foreground" />
}
