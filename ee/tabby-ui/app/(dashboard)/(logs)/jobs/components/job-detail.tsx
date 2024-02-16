'use client'

import { clearTimeout } from 'timers'
import React from 'react'
import { useSearchParams } from 'next/navigation'
import { useTheme } from 'next-themes'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { coldarkDark } from 'react-syntax-highlighter/dist/cjs/styles/prism'
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
    <div className="flex min-h-full flex-col">
      {fetching ? (
        <ListSkeleton />
      ) : (
        <div className="flex flex-1 flex-col gap-2">
          <JobsTable jobs={edges?.slice(0, 1)} shouldRedirect={false} />
          <Tabs defaultValue="stdout">
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
            <TabsContent value="stdout">
              <StdoutView value={currentNode?.stdout} />
            </TabsContent>
            <TabsContent value="stderr">
              <StdoutView value={currentNode?.stderr} />
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
  value,
  ...rest
}: React.HTMLAttributes<HTMLDivElement> & { value?: string }) {
  const { theme } = useTheme()
  return (
    <div
      className={cn(
        'min-h-80 mt-2 w-full overflow-x-hidden rounded-lg border bg-gray-50 dark:bg-gray-800',
        className
      )}
      {...rest}
    >
      {value ? (
        <SyntaxHighlighter
          wrapLongLines
          language="bash"
          style={theme === 'dark' ? coldarkDark : undefined}
          PreTag="div"
          customStyle={{
            margin: 0,
            background: 'transparent'
          }}
          codeTagProps={{
            style: {
              fontSize: '0.9rem',
              fontFamily: 'var(--font-mono)'
            }
          }}
        >
          {value ?? ''}
        </SyntaxHighlighter>
      ) : (
        <div className="p-4 font-mono text-[0.9rem]">No Data</div>
      )}
    </div>
  )
}
