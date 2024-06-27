'use client'

import React, { Suspense } from 'react'
import useSWRImmutable from 'swr/immutable'

import { GrepFile, RepositoryKind } from '@/lib/gql/generates/graphql'
import fetcher from '@/lib/tabby/fetcher'

import { SourceCodeSearchResult } from './source-code-search-result'
import {
  encodeURIComponentIgnoringSlash,
  getProviderVariantFromKind
} from './utils'

export interface SourceCodeSearchResult extends GrepFile {
  blob: string
}

interface SourceCodeSearchResultsProps {
  results?: GrepFile[]
  repoId?: string
  repositoryKind: RepositoryKind
}

export const SourceCodeSearchResults = ({
  ...props
}: SourceCodeSearchResultsProps) => {
  const currentURL = new URL(window.location.href)

  // FIXME: this feels semi fragile somehow?
  const query = currentURL.searchParams.get('q') ?? ''
  /**
   *
   */
  const [results, setResults] = React.useState<SourceCodeSearchResult[]>()

  /**
   * Match count. The tally of the lines across all results.
   */
  const [matchCount, setMatchCount] = React.useState<number>(0)

  const pathNamesToFetch = props.results?.map(result => result.path)

  // TODO: Share this globally
  const urls = pathNamesToFetch?.map(path =>
    encodeURIComponentIgnoringSlash(
      `/repositories/${getProviderVariantFromKind(props.repositoryKind)}/${
        props.repoId
      }/resolve/${path}`
    )
  )

  // TODO: Share this globally
  const multiFetcher = (urls: string[]) => {
    return Promise.all(
      urls.map(url => {
        return fetcher(url, {
          responseFormatter: async response => {
            if (!response.ok) return undefined
            const blob = await response.blob().then(blob => blob.text())
            return blob
          }
        })
      })
    )
  }

  const { data } = useSWRImmutable(urls, multiFetcher)

  React.useEffect(() => {
    if (data) {
      const newResults = props.results?.map((result, index) => {
        return {
          ...result,
          blob: data[index]
        }
      })
      setResults(newResults)
    } else {
      setResults([])
    }
  }, [data, props.results])

  React.useEffect(() => {
    setMatchCount(
      results?.reduce((acc, result) => {
        return (
          acc + result.lines.filter(line => line.subMatches.length > 0).length
        )
      }, 0) ?? 0
    )
  }, [results])

  return (
    <>
      <div className="mt-5 mb-7">
        <h1 className="font-semibold">Results for “{query}”</h1>
        <p className="text-sm text-muted-foreground">
          {matchCount} {matchCount === 1 ? 'match' : 'matches'}
        </p>
      </div>
      {results && results.length > 0 && (
        <ol className="grid gap-8">
          {results.map((result, i) => (
            // FIXME: This key should be unique
            <li key={i} className="">
              <SourceCodeSearchResult result={result} query={query} />
            </li>
          ))}
        </ol>
      )}
    </>
  )
}
