'use client'

import React, { use, useEffect, useState } from 'react'
import { set } from 'date-fns'
import useSWRImmutable from 'swr/immutable'

import { GrepFile, RepositoryKind } from '@/lib/gql/generates/graphql'
import fetcher from '@/lib/tabby/fetcher'

import {
  encodeURIComponentIgnoringSlash,
  getProviderVariantFromKind
} from '../utils'
import { GlobalSearchResult } from './result'

interface GlobalSearchResultsProps {
  results?: GrepFile[]
  query: string
  repoId?: string
  repositoryKind: RepositoryKind
}

export const GlobalSearchResults = ({ ...props }: GlobalSearchResultsProps) => {
  /**
   *
   */
  // TODO: Rename?
  const [results, setResults] = useState<any[]>() // FIXME: Any

  /**
   * Match count. The tally of the lines across all results.
   */
  const [matchCount, setMatchCount] = useState<number>(0)

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
  const multiFetcher = urls => {
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

  useEffect(() => {
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

  useEffect(() => {
    setMatchCount(
      results?.reduce((acc, result) => {
        return acc + result.lines.length
      }, 0)
    )
  }, [results])

  useEffect(() => {
    // TODO: Set a static query string based on the queryParam
    // to resolve the issue of the reactive h1
  })

  return (
    <>
      <div className="flex justify-between w-full items-start gap-16 my-4">
        {/* FIXME: This shouldn't update on type */}
        <div>
          <h1 className="text-xl font-semibold mb-0.5">
            Results for “{props.query}”
          </h1>
          <p>
            {matchCount} {matchCount === 1 ? 'match' : 'matches'}
          </p>
        </div>
        <button className="flex mt-1 gap-2 items-center shrink-0">
          {/* TODO: Use form component */}
          {/* TODO: Semantics */}
          <input type="checkbox" />
          <label className="text-sm">Ignore case</label>
        </button>
      </div>
      {results && results.length > 0 && (
        <ul className="grid gap-5">
          {results.map((result, i) => (
            // FIXME: This key should be unique
            <li key={i} className="">
              <GlobalSearchResult result={result} query={props.query} />
            </li>
          ))}
        </ul>
      )}
    </>
  )
}
