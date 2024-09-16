'use client'

import React from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import Ansi from '@curvenote/ansi-to-react'
import humanizerDuration from 'humanize-duration'
import moment from 'moment'
import { useQuery } from 'urql'

import { listJobRuns } from '@/lib/tabby/query'
import { cn } from '@/lib/utils'
import {
  IconChevronLeft,
  IconClock,
  IconHistory,
  IconSpinner,
  IconStopWatch
} from '@/components/ui/icons'
import { ListSkeleton } from '@/components/skeleton'

import { getJobDisplayName, getLabelByJobRun } from '../utils'

export default function JobRunDetail() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const id = searchParams.get('id')
  const [{ data, error, fetching }, reexecuteQuery] = useQuery({
    query: listJobRuns,
    variables: { ids: [id as string] },
    pause: !id
  })

  const currentNode = data?.jobRuns?.edges?.[0]?.node

  const stateLabel = getLabelByJobRun(currentNode)
  const isPending =
    (stateLabel === 'Pending' || stateLabel === 'Running') &&
    !currentNode?.stdout

  React.useEffect(() => {
    let timer: number
    if (currentNode?.createdAt && !currentNode?.finishedAt) {
      timer = window.setTimeout(() => {
        reexecuteQuery()
      }, 5000)
    }

    return () => {
      if (timer) {
        clearInterval(timer)
      }
    }
  }, [currentNode])

  return (
    <>
      {fetching ? (
        <ListSkeleton />
      ) : (
        <div className="flex flex-1 flex-col items-stretch gap-2">
          {currentNode && (
            <>
              <div
                onClick={() => router.back()}
                className="-ml-1 flex cursor-pointer items-center transition-opacity hover:opacity-60"
              >
                <IconChevronLeft className="mr-1 h-6 w-6" />
                <h2 className="scroll-m-20 text-3xl font-bold tracking-tight first:mt-0">
                  {getJobDisplayName(currentNode.job)}
                </h2>
              </div>
              <div className="mb-2 flex gap-x-5 text-sm text-muted-foreground lg:gap-x-10">
                <div className="flex items-center gap-1">
                  <IconStopWatch />
                  <p>State: {stateLabel}</p>
                </div>

                {currentNode.createdAt && (
                  <div className="flex items-center gap-1">
                    <IconClock />
                    <p>
                      Created:{' '}
                      {moment(currentNode.createdAt).format('YYYY-MM-DD HH:mm')}
                    </p>
                  </div>
                )}

                {currentNode.startedAt && (
                  <div className="flex items-center gap-1">
                    <IconClock />
                    <p>
                      Started:{' '}
                      {moment(currentNode.startedAt).format('YYYY-MM-DD HH:mm')}
                    </p>
                  </div>
                )}

                {currentNode.createdAt && currentNode.finishedAt && (
                  <div className="flex items-center gap-1">
                    <IconHistory />
                    <p>
                      Duration:{' '}
                      {humanizerDuration(
                        moment
                          .duration(
                            moment(currentNode.finishedAt).diff(
                              currentNode.startedAt
                            )
                          )
                          .asMilliseconds()
                      )}
                    </p>
                  </div>
                )}
              </div>
              <div className="flex flex-1 flex-col">
                <StdoutView value={currentNode?.stdout} pending={isPending} />
              </div>
            </>
          )}
        </div>
      )}
    </>
  )
}

function StdoutView({
  children,
  className,
  value,
  pending,
  ...rest
}: React.HTMLAttributes<HTMLDivElement> & {
  value?: string
  pending?: boolean
}) {
  return (
    <div
      className={cn(
        'relative mt-2 h-[72vh] w-full overflow-y-auto overflow-x-hidden rounded-lg border bg-gray-50 font-mono text-[0.9rem] dark:bg-gray-800',
        className
      )}
      {...rest}
    >
      {pending && !value && (
        <div className="absolute inset-0 flex items-center justify-center bg-background/60">
          <IconSpinner className="h-8 w-8" />
        </div>
      )}
      {value && (
        <pre className="whitespace-pre-wrap p-4">
          <Ansi>{value}</Ansi>
        </pre>
      )}
    </div>
  )
}
