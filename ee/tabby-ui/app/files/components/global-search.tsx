'use client'

import React, { FormEventHandler, useContext, useEffect, useState } from 'react'

import { graphql } from '@/lib/gql/generates'
import { GrepTextOrBase64, RepositoryKind } from '@/lib/gql/generates/graphql'
import { client } from '@/lib/tabby/gql'

import { GlobalSearchListItem } from './global-search/list-item'
import { SourceCodeBrowserContext } from './source-code-browser'
import { resolveRepositoryInfoFromPath } from './utils'

// TODO: Move these to a shared location

interface RepositoryGrep {
  kind: RepositoryKind
  id: string | number
  query: string
  rev?: string
}

interface GrepSubMatch {
  byteStart: number
  byteEnd: number
}

interface GrepLine {
  line: GrepTextOrBase64
  byteOffset: number
  lineNumber: number
  subMatches: GrepSubMatch[]
}

interface GrepFile {
  path: string
  lines: GrepLine[]
}

interface GlobalSearchProps {}

const globalSearchQuery = graphql(/* GraphQL */ `
  query GlobalSearch($id: ID!, $kind: RepositoryKind!, $query: String!) {
    repositoryGrep(kind: $kind, id: $id, query: $query) {
      path
      lines {
        line {
          text
          base64
        }
        byteOffset
        lineNumber
        subMatches {
          bytesStart
          bytesEnd
        }
      }
    }
  }
`)

const GlobalSearch: React.FC<GlobalSearchProps> = () => {
  const { activePath, activeRepo } = useContext(SourceCodeBrowserContext)

  const { repositoryKind } = resolveRepositoryInfoFromPath(activePath)

  const repoId = activeRepo?.id

  /**
   * The current search value. Set `onInput` or by the
   * setup effect when the URL has a query parameter.
   */
  const [value, setValue] = useState('')

  /**
   * Whether the search is currently running.
   */
  const [isRunning, setIsRunning] = useState(false)

  /**
   * The snippet(?) results of the search.
   * Set to the response of the `search` task.
   */
  const [results, setResults] = useState<GrepFile[] | null>(null)

  /**
   * Check if the URL has a query parameter and conditionally
   * set the value of the search input.
   */
  useEffect(() => {
    const urlParams = new URLSearchParams(window.location.search)
    const query = urlParams.get('q')

    if (query) {
      setValue(query)
    }
  }, [])

  /**
   * The async task to fetch the search results from the server.
   * Runs with every input change. Sets the value of the results
   */
  const onInput: FormEventHandler<HTMLInputElement> = e => {
    const q = e.currentTarget.value
    setValue(q)
    void search(q)
  }

  /**
   * The action run when the input is focused.
   * Checks for the case where the input has a value but no results,
   * such as when the input is populated from a URL parameter.
   */
  const onFocus = () => {
    if (value && !results) {
      void search(value)
    }
  }

  /**
   * The async task to fetch the search results from the server.
   * Called by the `onInput` event handler when the input value changes.
   */
  const search = async (query: string) => {
    setIsRunning(true)

    try {
      const { data } = (await client
        .query(globalSearchQuery, {
          id: repoId as string,
          kind: repositoryKind as RepositoryKind,
          query,
          pause: !repoId || !repositoryKind
        })
        // TODO: Fix types
        .toPromise()) as unknown as { data: { repositoryGrep: GrepFile[] } }

      setResults(data.repositoryGrep)
    } catch (e) {
      //  TODO: Handle error
      console.error('SEARCH ERROR', e)
    } finally {
      setIsRunning(false)
    }
  }

  return (
    <>
      <div className="w-full relative z-20">
        <input
          type="text"
          placeholder="Jump to a file or search the repository..."
          // Placeholder styles
          className="w-full h-9 pl-10  border border-gray-300 rounded"
          value={value}
          onInput={onInput}
          onFocus={onFocus}
        />
        <div className="absolute w-4 h-4 leading-none left-3 top-1/2 -translate-y-1/2">
          {isRunning ? '🌀' : '🔍'}
        </div>
        {results && results.length > 0 && (
          <div className="absolute translate-y-full bottom-0 left-0 w-full py-8 px-9 bg-white drop-shadow-2xl">
            <ol className="grid gap-2  overflow-hidden">
              {results.slice(0, 8).map((file, i) => {
                return (
                  <GlobalSearchListItem
                    key={i}
                    repoId={repoId as string}
                    repoKind={repositoryKind as RepositoryKind}
                    file={file}
                  />
                )
              })}
            </ol>
          </div>
        )}
      </div>
    </>
  )
}

export { GlobalSearch }
