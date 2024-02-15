'use client'

import { clearTimeout } from 'timers'
import React from 'react'
import { useSearchParams } from 'next/navigation'
import { useQuery } from 'urql'

import { listJobRuns } from '@/lib/tabby/query'
import { cn } from '@/lib/utils'
import { IconAlertTriangle, IconTerminalSquare } from '@/components/ui/icons'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { ListSkeleton } from '@/components/skeleton'

import { JobsTable } from './jobs-table'

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
      clearTimeout(timer)
    }
  }, [currentNode])

  return (
    <div>
      {fetching ? (
        <ListSkeleton />
      ) : (
        <div className="flex flex-col gap-2">
          <JobsTable jobs={edges?.slice(0, 1)} showOperation={false} />
          <Tabs defaultValue="stdout">
            <TabsList className="grid w-[400px] grid-cols-2">
              <TabsTrigger value="stdout">
                <IconTerminalSquare className="mr-1" />
                stdout
              </TabsTrigger>
              <TabsTrigger value="stderr" className="mr-1">
                <IconAlertTriangle />
                stderr
              </TabsTrigger>
            </TabsList>
            <TabsContent value="stdout">
              <StdoutView>{edges?.[0]?.node?.stdout}</StdoutView>
            </TabsContent>
            <TabsContent value="stderr">
              <StdoutView>{edges?.[0]?.node?.stderr}</StdoutView>
            </TabsContent>
          </Tabs>
        </div>
      )}
    </div>
  )
}

function StdoutView({
  children,
  className,
  ...rest
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn('rounded-lg border w-full p-2 mt-2', className)}
      {...rest}
    >
      <pre className="whitespace-pre">
        {children ? children : <div>No Data</div>}
      </pre>
    </div>
  )
}
