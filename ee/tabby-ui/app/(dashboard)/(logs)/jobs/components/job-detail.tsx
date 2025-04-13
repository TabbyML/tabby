'use client'

import React, { useEffect, useMemo, useRef, useState } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import Ansi from '@curvenote/ansi-to-react'
import humanizerDuration from 'humanize-duration'
import { concat, unionWith } from 'lodash-es'
import moment from 'moment'
import useSWRImmutable from 'swr/immutable'
import { useQuery } from 'urql'

import { useLatest } from '@/lib/hooks/use-latest'
import fetcher from '@/lib/tabby/fetcher'
import { listJobRuns } from '@/lib/tabby/query'
import { cn } from '@/lib/utils'
import {
  IconChevronLeft,
  IconClock,
  IconHistory,
  IconSpinner,
  IconStopWatch
} from '@/components/ui/icons'
import { LoadMoreIndicator } from '@/components/load-more-indicator'
import { ListSkeleton } from '@/components/skeleton'

import { getJobDisplayName, getLabelByJobRun } from '../utils'

interface LogChunk {
  startByte: number
  endByte: number
  logs: string
}

const CHUNK_SIZE = 50 * 1000

export default function JobRunDetail() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const id = searchParams.get('id')
  const [chunks, setChunks] = useState<LogChunk[]>([])
  const [loadedBytes, setLoadedBytes] = useState(-1)
  const [totalBytes, setTotalBytes] = useState<number | undefined>()

  const [{ data, fetching }, reexecuteQuery] = useQuery({
    query: listJobRuns,
    variables: { ids: [id as string] },
    pause: !id
  })
  const currentNode = data?.jobRuns?.edges?.[0]?.node
  const endCursor = useRef<number | undefined>()
  const shouldFetchLogs = !!id && !fetching && !currentNode?.stdout
  const {
    data: logsData,
    mutate,
    isLoading
  } = useSWRImmutable(
    id ? [`/background-jobs/${id}/logs`, loadedBytes + 1] : null,
    ([url, start]) => {
      return fetcher(url, {
        headers: {
          Range: `bytes=${start}-${start + CHUNK_SIZE}`
        },
        responseFormatter: res => {
          const contentRange = res.headers.get('Content-Range')
          if (!contentRange) return null
          if (res.status === 206) {
            const [range, total] = contentRange
              .replace(/^bytes\s/, '')
              .split('/')
            const [start, end] = range?.split('-')
            endCursor.current = parseInt(end)
            return res.text().then(text => ({
              logs: text,
              totalBytes: parseInt(total),
              startByte: parseInt(start),
              endByte: parseInt(end)
            }))
          }
        },
        errorHandler: response => {
          throw new Error(response?.statusText.toString())
        }
      })
    }
  )

  // join logs
  useEffect(() => {
    if (logsData) {
      setChunks(prev => {
        return unionWith(concat(prev, logsData), (x, y) => {
          return x.startByte === y.startByte && x.endByte === y.endByte
        })
      })
      setTotalBytes(logsData?.totalBytes ?? 0)
    }
  }, [logsData])

  const finalLogs = useMemo(() => {
    if (currentNode?.stdout) return currentNode?.stdout
    const logs = chunks.reduce((sum, cur) => sum + cur.logs, '')
    return processPartialLine(logs)
  }, [currentNode?.stdout, chunks])

  const stateLabel = getLabelByJobRun(currentNode)
  const isPending =
    (stateLabel === 'Pending' || stateLabel === 'Running') && !finalLogs

  const isLoadedCompleted =
    !!totalBytes && chunks[chunks.length - 1]?.endByte >= totalBytes - 1

  const handleBackNavigation = () => {
    if (typeof window !== 'undefined' && window.history.length <= 1) {
      router.push('/jobs')
    } else {
      router.back()
    }
  }

  const loadMore = useLatest(() => {
    if (isLoading) return

    if (endCursor.current) {
      const nextCursor = endCursor.current
      // setLoadedBytes to trigger fetch request
      setTimeout(() => {
        setLoadedBytes(nextCursor)
      }, 100)
    }
  })

  React.useEffect(() => {
    let timer: number
    if (currentNode?.createdAt && !currentNode?.finishedAt) {
      timer = window.setTimeout(() => {
        reexecuteQuery()
        if (isLoadedCompleted || !chunks.length) {
          mutate()
        }
      }, 5000)
    }

    return () => {
      if (timer) {
        clearInterval(timer)
      }
    }
  }, [currentNode, isLoadedCompleted, chunks?.length])

  return (
    <>
      {fetching ? (
        <ListSkeleton />
      ) : (
        <div className="flex flex-1 flex-col items-stretch gap-2">
          {currentNode && (
            <>
              <div
                onClick={handleBackNavigation}
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
                <StdoutView value={finalLogs} pending={isPending}>
                  {shouldFetchLogs && !isLoadedCompleted && (
                    <LoadMoreIndicator
                      intersectionOptions={{
                        trackVisibility: true,
                        delay: 200,
                        rootMargin: '100px 0px 0px 0px'
                      }}
                      onLoad={loadMore.current}
                      isFetching={isLoading}
                    >
                      <div className="flex justify-center">
                        <IconSpinner />
                      </div>
                    </LoadMoreIndicator>
                  )}
                </StdoutView>
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
      {!!value && (
        <>
          <pre className="whitespace-pre-wrap p-4">
            <Ansi>{value}</Ansi>
          </pre>
          {children}
        </>
      )}
    </div>
  )
}

function processPartialLine(logs: string) {
  if (!logs) return logs

  const lines = logs.split('\n')
  const lastLine = lines[lines.length - 1]
  const hasLineBreak = lastLine.endsWith('\n')

  return hasLineBreak ? lines.join('\n') : lines.slice(0, -1).join('\n')
}
