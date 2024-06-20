'use client'

import React, { useEffect, useState } from 'react'
import useSWRImmutable from 'swr/immutable'

import { GrepFile, RepositoryKind } from '@/lib/gql/generates/graphql'
import fetcher from '@/lib/tabby/fetcher'

import {
  encodeURIComponentIgnoringSlash,
  getProviderVariantFromKind
} from '../utils'
import { GlobalSearchResult } from './result'

interface GlobalSearchResultsProps {
  results: GrepFile[] | null
  repoId: string
  repositoryKind: RepositoryKind
  hidePopover: () => void
}

export const GlobalSearchResults = ({ ...props }: GlobalSearchResultsProps) => {
  const [results, setResults] = useState<any[] | null>(null)

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
      if (newResults) {
        setResults(newResults)
      }
    }
  }, [data, props.results])

  return (
    <>
      {results && (
        <>
          {results.map((result, i) => (
            <GlobalSearchResult
              key={i}
              result={result}
              hidePopover={props.hidePopover}
            />
          ))}
        </>
      )}
    </>
  )
}
