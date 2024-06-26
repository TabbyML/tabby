import React from 'react'
import Link from 'next/link'
import moment from 'moment'

import { Button } from '@/components/ui/button'
import { IconCirclePlay, IconSpinner } from '@/components/ui/icons'
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger
} from '@/components/ui/tooltip'

interface JobInfoProps {
  jobInfo: {
    command: string
    lastJobRun?: {
      id: string
      job: string
      createdAt: any
      finishedAt?: any | null
      exitCode?: number | null
    } | null
  }
  onTrigger: () => Promise<any>
}

function JobTrigger({
  onTrigger,
  isPending
}: Pick<JobInfoProps, 'onTrigger'> & { isPending?: boolean }) {
  const [loading, setLoading] = React.useState(false)
  const handleTrigger = () => {
    const res = onTrigger()

    if (res && res instanceof Promise) {
      setLoading(true)
      res.finally(() => setLoading(false))
    }

    return res
  }

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button
          size="icon"
          variant="ghost"
          onClick={handleTrigger}
          disabled={loading || isPending}
        >
          {loading || isPending ? (
            <IconSpinner />
          ) : (
            <IconCirclePlay strokeWidth={1} className="h-5 w-5" />
          )}
        </Button>
      </TooltipTrigger>
      <TooltipContent>
        <p>Run</p>
      </TooltipContent>
    </Tooltip>
  )
}

function LastJobRunInfo({ jobInfo }: Pick<JobInfoProps, 'jobInfo'>) {
  if (!jobInfo?.lastJobRun) return null

  return (
    <Link
      href={`/jobs/detail?id=${jobInfo.lastJobRun.id}`}
      className="flex items-center gap-1 underline hover:text-foreground/50"
    >
      {moment(jobInfo.lastJobRun.createdAt).format('YYYY-MM-DD HH:mm')}
    </Link>
  )
}

export function JobInfoView(props: JobInfoProps) {
  const { jobInfo, onTrigger } = props
  const isJobPending =
    !!jobInfo?.lastJobRun && jobInfo.lastJobRun.exitCode === null

  return (
    <div className="flex flex-col items-center gap-1 lg:flex-row">
      <LastJobRunInfo jobInfo={jobInfo} />
      <JobTrigger onTrigger={onTrigger} isPending={isJobPending} />
    </div>
  )
}
