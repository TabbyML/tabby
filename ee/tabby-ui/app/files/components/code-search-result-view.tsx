'use client'

import React from 'react'
import { useSearchParams } from 'next/navigation'

import { GrepFile } from '@/lib/gql/generates/graphql'
import { ListSkeleton } from '@/components/skeleton'

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

  const matchCount: number = React.useMemo(() => {
    let count = 0
    if (!props.results) return count

    for (const result of props.results) {
      const curCount = result.lines.reduce((sum, cur) => {
        const _matchCount = cur.subMatches.length
        return sum + _matchCount
      }, 0)

      count += curCount
    }
    return count
  }, [props.results])

  return (
    <>
      <div className="mt-5">
        <h1 className="font-semibold">Results for “{query}”</h1>
      </div>
      {props.loading ? (
        <ListSkeleton className="mt-2" />
      ) : (
        <>
          <p className="mb-7 text-sm text-muted-foreground">
            {matchCount} {matchCount === 1 ? 'match' : 'matches'}
          </p>
          {results && results.length > 0 && (
            <ol className="grid gap-8">
              {results.map((result, i) => (
                <li key={`${result.path}-${i}`}>
                  <SourceCodeSearchResult result={result} query={query} />
                </li>
              ))}
            </ol>
          )}
        </>
      )}
    </>
  )
}
