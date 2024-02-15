'use client'

import React from 'react'
import { useSearchParams } from 'next/navigation'
import { useQuery } from 'urql'

import { DEFAULT_PAGE_SIZE } from '@/lib/constants'
import { ListJobRunsQueryVariables } from '@/lib/gql/generates/graphql'
import { listJobRuns } from '@/lib/tabby/query'
import { cn } from '@/lib/utils'
import { IconAlertTriangle, IconTerminalSquare } from '@/components/ui/icons'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { ListSkeleton } from '@/components/skeleton'

import { JobsTable } from './jobs-table'

const PAGE_SIZE = DEFAULT_PAGE_SIZE
export default function JobRunDetail() {
  const searchParams = useSearchParams()
  const id = searchParams.get('id')
  const [variables, setVariables] = React.useState<ListJobRunsQueryVariables>({
    last: PAGE_SIZE
  })
  const [{ data, error, fetching }] = useQuery({
    query: listJobRuns,
    variables,
    pause: !id
  })

  const edges = data?.jobRuns?.edges

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
