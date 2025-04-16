'use client'

import React, { useEffect, useMemo, useRef, useState } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import Ansi from '@curvenote/ansi-to-react'
import humanizerDuration from 'humanize-duration'
import { concat, sortBy, unionWith } from 'lodash-es'
import moment from 'moment'
import useSWR from 'swr'
import { useQuery } from 'urql'

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

const CHUNK_SIZE = 20 * 1000

export default function JobRunDetail() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const id = searchParams.get('id')
  const [chunks, setChunks] = useState<LogChunk[]>([])
  const cursor = useRef<number>(0)
  const totalBytes = useRef<number>(0)
  const [startBytes, setStartBytes] = useState(-1)
  const fetchLogEndPoint = id ? `/background-jobs/${id}/logs` : null
  const [{ data, fetching: fetchingJobNode }, reexecuteQuery] = useQuery({
    query: listJobRuns,
    variables: { ids: [id as string] },
    pause: !id
  })
  const currentNode = data?.jobRuns?.edges?.[0]?.node
  const shouldFetchLogs = !!id && !fetchingJobNode && !currentNode?.stdout
  const {
    data: logsData,
    mutate,
    isLoading: isLoadingLogs,
    isValidating: isValidatingLogs
  } = useSWR(
    fetchLogEndPoint ? [fetchLogEndPoint, startBytes + 1] : null,
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
            cursor.current = parseInt(end)
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
    },
    {
      errorRetryCount: 1
    }
  )

  const isFetchingLogs = isLoadingLogs || isValidatingLogs

  // join logs
  useEffect(() => {
    if (logsData) {
      const newChunk: LogChunk = {
        startByte: logsData.startByte,
        endByte: logsData.endByte,
        logs: logsData.logs
      }
      setChunks(prev => {
        return sortBy(
          unionWith(concat([newChunk], prev), (x, y) => {
            return x.startByte === y.startByte
          }),
          'startByte'
        )
      })
      totalBytes.current = logsData?.totalBytes ?? 0
      // setTotalBytes(logsData?.totalBytes ?? 0)
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
    !!totalBytes.current &&
    chunks[chunks.length - 1]?.endByte >= totalBytes.current - 1

  const handleBackNavigation = () => {
    if (typeof window !== 'undefined' && window.history.length <= 1) {
      router.push('/jobs')
    } else {
      router.back()
    }
  }

  const handleLoadMore = () => {
    if (isFetchingLogs) return
    if (cursor.current && cursor.current + 1 < totalBytes.current) {
      const nextCursor = cursor.current
      setStartBytes(nextCursor)
    }
  }

  React.useEffect(() => {
    let timer: number
    if (currentNode?.createdAt && !currentNode?.finishedAt) {
      timer = window.setTimeout(() => {
        reexecuteQuery()
        if (!isFetchingLogs && (isLoadedCompleted || !chunks.length)) {
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
      {fetchingJobNode ? (
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
                        delay: 100,
                        rootMargin: '200px 0px 0px 0px'
                      }}
                      itemCount={chunks.length}
                      onLoad={handleLoadMore}
                      isFetching={isFetchingLogs}
                    >
                      <div className="my-8 flex justify-center">
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
