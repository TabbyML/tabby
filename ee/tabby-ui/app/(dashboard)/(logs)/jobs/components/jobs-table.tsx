import Link from 'next/link'
import moment from 'moment'

import { ListJobRunsQuery } from '@/lib/gql/generates/graphql'
import { Badge } from '@/components/ui/badge'
import { buttonVariants } from '@/components/ui/button'
import { IconFileSearch } from '@/components/ui/icons'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow
} from '@/components/ui/table'

interface JobsTableProps {
  jobs: ListJobRunsQuery['jobRuns']['edges'] | undefined
  showOperation?: boolean
}

export const JobsTable: React.FC<JobsTableProps> = ({
  jobs,
  showOperation = true
}) => {
  return (
    <Table>
      <TableHeader className="sticky top-0">
        <TableRow>
          <TableHead className="w-[200px]">Start Time</TableHead>
          <TableHead className="w-[100px]">Duration</TableHead>
          <TableHead className="w-[100px]">Job Type</TableHead>
          <TableHead className="w-[100px]">Exist Code</TableHead>
          {showOperation && (
            <TableHead className="w-[100px] text-right">Detail</TableHead>
          )}
        </TableRow>
      </TableHeader>
      <TableBody>
        {!jobs?.length ? (
          <TableRow>
            <TableCell
              colSpan={showOperation ? 5 : 4}
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
                  <TableCell>{x.node.exitCode}</TableCell>
                  {showOperation && (
                    <TableCell className="text-right">
                      <Link
                        href={`/jobs/detail?id=${x.node.id}`}
                        className={buttonVariants({ variant: 'ghost' })}
                      >
                        <IconFileSearch />
                      </Link>
                    </TableCell>
                  )}
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
