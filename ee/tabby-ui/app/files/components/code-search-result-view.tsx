'use client'

import React from 'react'
import { useSearchParams } from 'next/navigation'

import { GrepFile } from '@/lib/gql/generates/graphql'
import { cn } from '@/lib/utils'
import { Skeleton } from '@/components/ui/skeleton'

import { SourceCodeSearchResult } from './code-search-result'

export interface SourceCodeSearchResult extends GrepFile {
  blob: string
}

interface CodeSearchResultViewProps {
  results?: GrepFile[]
  loading?: boolean
}

export const CodeSearchResultView = (props: CodeSearchResultViewProps) => {
  const searchParams = useSearchParams()
  const query = searchParams.get('q')?.toString() ?? ''

  const results: SourceCodeSearchResult[] = React.useMemo(() => {
    const _results = props.results
    return (
      _results?.map(item => ({
        ...item,
        blob: item.lines.reduce((sum, cur) => {
          return sum + (cur.line.text ?? '')
        }, '')
      })) ?? []
    )
  }, [props.results])

  return (
    <>
      {props.loading ? (
        <CodeSearchSkeleton className="mt-3" />
      ) : (
        <>
          <h1 className="mb-2 mt-1 font-semibold">
            {results?.length || 0} files
          </h1>
          {results?.length > 0 ? (
            <>
              {results.map((result, i) => (
                <div key={`${result.path}-${i}`}>
                  <SourceCodeSearchResult result={result} query={query} />
                </div>
              ))}
            </>
          ) : (
            // FIXME
            <div>not found</div>
          )}
        </>
      )}
    </>
  )
}

function CodeSearchSkeleton({ className }: React.ComponentProps<'div'>) {
  return (
    <div className={cn('flex flex-col gap-3', className)}>
      <Skeleton className="h-4 w-[20%]" />
      <Skeleton className="h-4 w-full" />
      <Skeleton className="h-8 w-full" />
    </div>
  )
}
