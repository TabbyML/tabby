'use client'

import React from 'react'
import { useSearchParams } from 'next/navigation'
import humanizerDuration from 'humanize-duration'
import numeral from 'numeral'

import { GrepFile } from '@/lib/gql/generates/graphql'
import { cn } from '@/lib/utils'
import { Skeleton } from '@/components/ui/skeleton'

import { SourceCodeSearchResult } from './code-search-result'

export interface SourceCodeSearchResult extends GrepFile {
  blob: string
}

interface CodeSearchResultViewProps {
  results?: GrepFile[]
  requestDuration?: number
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

  const matchCount: string = React.useMemo(() => {
    let count = 0
    if (!props.results) return '0'

    for (const result of props.results) {
      const curCount = result.lines.reduce((sum, cur) => {
        const _matchCount = cur.subMatches.length
        return sum + _matchCount
      }, 0)

      count += Math.max(curCount, 1)
    }
    const format = count < 1000 ? '0' : '0.0a'
    return numeral(count).format(format)
  }, [props.results])

  const duration = humanizerDuration.humanizer({
    units: ['d', 'h', 'm', 's'],
    spacer: '',
    maxDecimalPoints: 2,
    language: 'shortEn',
    languages: {
      shortEn: {
        m: () => 'm',
        s: () => 's'
      }
    }
  })(props.requestDuration ?? 0)

  return (
    <>
      {props.loading ? (
        <CodeSearchSkeleton className="mt-3" />
      ) : (
        <>
          <h1 className="sticky top-0 z-20 bg-background pb-2 pt-1 font-semibold">
            {matchCount} results in {duration}
          </h1>
          {results?.map((result, i) => (
            <div key={`${result.path}-${i}`}>
              <SourceCodeSearchResult result={result} query={query} />
            </div>
          ))}
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
