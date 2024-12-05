import React from 'react'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import moment from 'moment'

import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { IconCirclePlay, IconSpinner } from '@/components/ui/icons'
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger
} from '@/components/ui/tooltip'

interface JobInfoProps {
  jobInfo:
    | {
        command: string
        lastJobRun?: {
          id: string
          job: string
          createdAt: any
          finishedAt?: any | null
          exitCode?: number | null
        } | null
      }
    | undefined
    | null
  onTrigger: () => Promise<any>
  className?: string
}

function JobTrigger({
  onTrigger,
  isPending,
  jobLink
}: Pick<JobInfoProps, 'onTrigger'> & {
  isPending?: boolean
  jobLink?: string
}) {
  const router = useRouter()
  const [loading, setLoading] = React.useState(false)
  const handleClick = () => {
    if (isPending) {
      if (jobLink) {
        router.push(jobLink)
      }
      return
    }

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
          onClick={handleClick}
          disabled={loading}
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

function LastJobRunInfo({
  jobInfo,
  className
}: Pick<JobInfoProps, 'jobInfo'> & { className?: string }) {
  if (!jobInfo?.lastJobRun) return null

  return (
    <Link
      href={`/jobs/detail?id=${jobInfo.lastJobRun.id}`}
      className={cn(
        'flex items-center gap-1 underline hover:text-foreground/50',
        className
      )}
    >
      {moment(jobInfo.lastJobRun.createdAt).format('YYYY-MM-DD HH:mm')}
    </Link>
  )
}

export function JobInfoView(props: JobInfoProps) {
  const { jobInfo, onTrigger, className } = props
  const isJobPending =
    !!jobInfo?.lastJobRun && jobInfo.lastJobRun.exitCode === null
  const jobLink = jobInfo?.lastJobRun?.id
    ? `/jobs/detail?id=${jobInfo.lastJobRun.id}`
    : undefined

  return (
    <div className={cn('flex items-center gap-1', className)}>
      <LastJobRunInfo jobInfo={jobInfo} className="hidden lg:block" />
      <JobTrigger
        onTrigger={onTrigger}
        isPending={isJobPending}
        jobLink={jobLink}
      />
    </div>
  )
}
