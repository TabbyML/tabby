'use client'

import React from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import Ansi from '@curvenote/ansi-to-react'
import humanizerDuration from 'humanize-duration'
import moment from 'moment'
import { useQuery } from 'urql'

import { listJobRuns } from '@/lib/tabby/query'
import {
  IconChevronLeft,
  IconClock,
  IconHistory,
  IconStopWatch
} from '@/components/ui/icons'
import { ListSkeleton } from '@/components/skeleton'
import { getJobDisplayName, getLabelByJobRun } from '../utils'

import "@patternfly/react-core/dist/styles/base.css";
import { LogViewer, LogViewerSearch } from '@patternfly/react-log-viewer';
import { Toolbar, ToolbarContent, ToolbarItem } from '@patternfly/react-core';

const BasicSearchLogViewer = ({data}: {data?: string}) => {
  return (
    <LogViewer
      data={data}
      toolbar={
        <Toolbar>
          <ToolbarContent>
            <ToolbarItem>
              <LogViewerSearch placeholder="Search value" />
            </ToolbarItem>
          </ToolbarContent>
        </Toolbar>
      }
    />
  )
};

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
                <BasicSearchLogViewer data={currentNode?.stdout} />
              </div>
            </>
          )}
        </div>
      )}
    </>
  )
}
