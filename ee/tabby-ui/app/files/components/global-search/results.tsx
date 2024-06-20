'use client'

import React, { useEffect, useState } from 'react'
import useSWRImmutable from 'swr/immutable'

import { GrepFile, GrepLine, RepositoryKind } from '@/lib/gql/generates/graphql'
import fetcher from '@/lib/tabby/fetcher'

import {
  encodeURIComponentIgnoringSlash,
  getProviderVariantFromKind
} from '../utils'
import { GlobalSearchResult } from './result'

interface GlobalSearchResultsProps {
  results: GrepFile[] | null
  repoId?: string
  repositoryKind?: RepositoryKind
  hidePopover: () => void
}

export const GlobalSearchResults = ({ ...props }: GlobalSearchResultsProps) => {
  const [results, setResults] = useState<any[] | null>(null)

  const pathNamesToFetch = props.results?.map(result => result.path)

  const urls = pathNamesToFetch?.map(path =>
    encodeURIComponentIgnoringSlash(
      `/repositories/${getProviderVariantFromKind(props.repositoryKind)}/${
        props.repoId
      }/resolve/${path}`
    )
  )

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

  // go through the results and add the blob to the object

  useEffect(() => {
    if (data) {
      const newResults = props.results?.map((result, index) => {
        return {
          ...result,
          blob: data[index]
        }
      })
      if (newResults) {
        setResults(newResults)
      }
    }
  }, [data, props.results])

  if (results) {
    return (
      <div>
        {results.map((result, i) => (
          <GlobalSearchResult
            key={i}
            result={result}
            hidePopover={props.hidePopover}
          />
        ))}
      </div>
    )
  }

  // What we're passing to the ListItem must be a loaded File

  return <></>
}
