'use client'

import React from 'react'
import { useSearchParams } from 'next/navigation'
import Ansi from '@curvenote/ansi-to-react'
import humanizerDuration from 'humanize-duration'
import moment from 'moment'
import { useQuery } from 'urql'
import { isNil } from 'lodash-es'

import { listJobRuns } from '@/lib/tabby/query'
import { cn } from '@/lib/utils'
import { IconAlertTriangle, IconTerminalSquare } from '@/components/ui/icons'
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
              <h1 className="text-4xl font-semibold tracking-tight first:mt-0">
                {currentNode.job}
              </h1>
              <div className="flex gap-16 pb-6 pt-2">
                <div>
                  <p
                    className={cn('font-bold', {
                      'text-orange-400': isNil(currentNode.exitCode),
                      'text-green-400': currentNode.exitCode === 0,
                      'text-red-400': currentNode.exitCode === 1
                    })}
                  >
                    {getLabelByExitCode(currentNode.exitCode)}
                  </p>
                  <p className="text-sm text-muted-foreground">Status</p>
                </div>

                {currentNode.createdAt && (
                  <div>
                    <p>
                      {moment(currentNode.createdAt).format(
                        'MMMM D, YYYY h:mm a'
                      )}
                    </p>
                    <p className="text-sm text-muted-foreground">Started At</p>
                  </div>
                )}

                {currentNode.createdAt && currentNode.finishedAt && (
                  <div>
                    <p>
                      {humanizerDuration(
                        moment
                          .duration(
                            moment(currentNode.finishedAt).diff(
                              currentNode.createdAt
                            )
                          )
                          .asMilliseconds()
                      )}
                    </p>
                    <p className="text-sm text-muted-foreground">Duration</p>
                  </div>
                )}
              </div>
              {/* <JobsTable jobs={edges?.slice(0, 1)} shouldRedirect={false} /> */}
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
