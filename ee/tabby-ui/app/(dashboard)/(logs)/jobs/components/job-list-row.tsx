import Link from 'next/link'
import moment from 'moment'
import humanizerDuration from 'humanize-duration'

import { cn } from '@/lib/utils'
import { findColorByExitCode, findLabelByExitCode } from '../utils/state'

import {
  TableCell,
  TableRow
} from '@/components/ui/table'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip"

import type { ListJobRunsQuery } from '@/lib/gql/generates/graphql'

type TJobRun = ListJobRunsQuery['jobRuns']['edges'][0]

export default function JobListRow ({
  name,
  jobs
}: {
  name: string;
  jobs?: TJobRun[] 
}) {
  if (!jobs) return <></>

  const lastJob = jobs[0]
  const lastFinishedJob = jobs?.find(job => Boolean(job.node.finishedAt))
  const currentStatte = findLabelByExitCode(lastJob.node.exitCode)
  const lastSuccessAt = lastFinishedJob
    ? moment(lastFinishedJob.node.finishedAt).format('MMMM D, YYYY h:mm a')
    : null
  return (
    <TableRow>
      <TableCell className="font-bold">
        {name}
      </TableCell>
      <TableCell>
        {currentStatte}
      </TableCell>
      <TableCell>
        <div className="flex">
          {jobs?.map(job => {
            const { createdAt, finishedAt } = job.node
            const startAt = createdAt && moment(createdAt).format('MMMM D, YYYY h:mm a')
            const duration = createdAt && finishedAt && humanizerDuration(moment
              .duration(moment(finishedAt).diff(createdAt))
              .asMilliseconds())
            const color = findColorByExitCode(job.node.exitCode)
            return (
              <TooltipProvider delayDuration={0} key={job.node.id}>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Link
                      href={`/jobs/detail?id=${job.node.id}`}
                      className={cn('mr-1 h-8 w-2 rounded-full hover:opacity-70', `bg-${color}`)} />
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
        {lastSuccessAt}
      </TableCell>
    </TableRow>
  )
}