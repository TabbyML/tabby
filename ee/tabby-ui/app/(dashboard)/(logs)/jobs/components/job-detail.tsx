'use client'

import React from 'react'
import { useSearchParams } from 'next/navigation'
import Ansi from '@curvenote/ansi-to-react'
import humanizerDuration from 'humanize-duration'
import moment from 'moment'
import { useQuery } from 'urql'

import { listJobRuns } from '@/lib/tabby/query'
import { cn } from '@/lib/utils'
import { IconAlertTriangle, IconTerminalSquare, IconHistory, IconRotate, IconClock } from '@/components/ui/icons'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { ListSkeleton } from '@/components/skeleton'

import { getLabelByExitCode } from '../utils/state'

export default function JobRunDetail() {
  const searchParams = useSearchParams()
  const id = searchParams.get('id')
  const [{ data, error, fetching }, reexecuteQuery] = useQuery({
    query: listJobRuns,
    variables: { ids: [id as string] },
    pause: !id
  })

  const edges = data?.jobRuns?.edges?.slice(0, 1)
  const currentNode = data?.jobRuns?.edges?.[0]?.node

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
              <h2 className="scroll-m-20 text-3xl font-bold tracking-tight first:mt-0">
                {currentNode.job}
              </h2>
              <div className="mb-8 flex gap-x-5 text-sm text-muted-foreground lg:gap-x-10">
                <div className="flex items-center gap-1">
                  <IconRotate />
                  <p>State: {getLabelByExitCode(currentNode.exitCode)}</p>
                </div>
                
                {currentNode.createdAt && (
                  <div className="flex items-center gap-1">
                    <IconClock />
                    <p>Started: {moment(currentNode.createdAt).format('YYYY-MM-DD HH:mm')}</p>
                  </div>
                )}

                {currentNode.createdAt && currentNode.finishedAt && (
                  <div className="flex items-center gap-1">
                    <IconHistory />
                    <p>Duration: {humanizerDuration(
                      moment
                        .duration(
                          moment(currentNode.finishedAt).diff(
                            currentNode.createdAt
                          )
                        )
                        .asMilliseconds()
                    )}</p>
                  </div>
                )}
              </div>
              <Tabs defaultValue="stdout" className="flex flex-1 flex-col">
                <TabsList className="grid w-[400px] grid-cols-2">
                  <TabsTrigger value="stdout">
                    <IconTerminalSquare className="mr-1" />
                    stdout
                  </TabsTrigger>
                  <TabsTrigger value="stderr">
                    <IconAlertTriangle className="mr-1" />
                    stderr
                  </TabsTrigger>
                </TabsList>
                <div className="flex flex-1 flex-col">
                  <TabsContent value="stdout">
                    <StdoutView value={currentNode?.stdout} />
                  </TabsContent>
                  <TabsContent value="stderr">
                    <StdoutView value={currentNode?.stderr} />
                  </TabsContent>
                </div>
              </Tabs>
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
  ...rest
}: React.HTMLAttributes<HTMLDivElement> & { value?: string }) {
  return (
    <div
      className={cn(
        'mt-2 h-[66vh] w-full overflow-y-auto overflow-x-hidden rounded-lg border bg-gray-50 font-mono text-[0.9rem] dark:bg-gray-800',
        className
      )}
      {...rest}
    >
      {value ? (
        <pre className="whitespace-pre-wrap p-4">
          <Ansi>{value}</Ansi>
        </pre>
      ) : (
        <div className="p-4">No Data</div>
      )}
    </div>
  )
}
