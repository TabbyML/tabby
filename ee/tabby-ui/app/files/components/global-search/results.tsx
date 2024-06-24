'use client'

import React, { useEffect, useState } from 'react'
import useSWRImmutable from 'swr/immutable'

import { GrepFile, RepositoryKind } from '@/lib/gql/generates/graphql'
import fetcher from '@/lib/tabby/fetcher'
import { Button } from '@/components/ui/button'

import {
  encodeURIComponentIgnoringSlash,
  getProviderVariantFromKind
} from '../utils'
import { GlobalSearchResult } from './result'

interface GlobalSearchResultsProps {
  results?: GrepFile[]
  query?: string
  repoId?: string
  repositoryKind?: RepositoryKind
  hidePopover: () => void
}

export const GlobalSearchResults = ({ ...props }: GlobalSearchResultsProps) => {
  const [results, setResults] = useState<any[]>()

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

  return (
    <>
      <div className="flex justify-between w-full  mb-4">
        {/* FIXME: This shouldn't update on type */}
        <h1 className="text-xl font-semibold mb-2">
          Results for “{props.query}”
        </h1>
        {/* TODO: Use form component */}
        <button className="flex gap-2 items-center">
          {/* TODO: Semantics */}
          <input type="checkbox" />
          <label className="text-sm">Ignore case</label>
        </button>
      </div>
      {results && results.length > 0 ? (
        <ul className="grid gap-5">
          {results.map((result, i) => (
            // FIXME: This key should be unique
            <li key={i} className="">
              <GlobalSearchResult
                result={result}
                hidePopover={props.hidePopover}
              />
            </li>
          ))}
        </ul>
      ) : (
        <p className="text-muted-foreground">No matches</p>
      )}
    </>
  )
}
