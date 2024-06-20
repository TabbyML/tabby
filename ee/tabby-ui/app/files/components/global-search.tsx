'use client'

import React, { FormEventHandler, useContext, useEffect, useState } from 'react'
import * as DropdownMenu from '@radix-ui/react-dropdown-menu'
import { PopoverContent, PopoverTrigger } from '@radix-ui/react-popover'

import { graphql } from '@/lib/gql/generates'
import { GrepTextOrBase64, RepositoryKind } from '@/lib/gql/generates/graphql'
import { client } from '@/lib/tabby/gql'
import { IconSearch, IconSpinner } from '@/components/ui/icons'
import { Popover } from '@/components/ui/popover'
import {
  SearchableSelect,
  SearchableSelectAnchor,
  SearchableSelectContent,
  SearchableSelectInput
} from '@/components/searchable-select'

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
  const [popoverIsShown, setPopoverIsShown] = useState(false)

  const { repositoryKind } = resolveRepositoryInfoFromPath(activePath)

  const repoId = activeRepo?.id

  /**
   * The current search value. Set `onInput` or by the
   * setup effect when the URL has a query parameter.
   */
  const [value, setValue] = useState('')

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
    if (q && !popoverIsShown) setPopoverIsShown(true)

    setValue(q)
    void search(q)
  }

  /**
   * The async task to fetch the search results from the server.
   * Called by the `onInput` event handler when the input value changes.
   */
  const search = async (query: string) => {
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
  }

  // FIXME: Currently the keys used to determine which option is highlighted are the same across results. for example, result 1 line 1 has an index of 1. But so does result 100 line 1, since the index is only based on the local array.

  return (
    <div className="px-4 py-2 w-full max-w-[600px]">
      <SearchableSelect
        options={undefined}
        open={popoverIsShown}
        onOpenChange={() => {
          if (value && results?.length) {
            setPopoverIsShown(true)
          } else {
            setPopoverIsShown(false)
          }
        }}
      >
        <SearchableSelectAnchor className="relative w-full">
          <SearchableSelectInput
            type="text"
            placeholder="Search the repository..."
            // Placeholder styles
            className="w-full h-9 pl-9 relative border border-border rounded"
            value={value}
            onChange={onInput}
          />
          <div className="absolute leading-none left-3 top-1/2 -translate-y-1/2 opacity-50">
            <IconSearch />
          </div>
        </SearchableSelectAnchor>
        <SearchableSelectContent
          sideOffset={4}
          alignOffset={-4}
          autoFocus={false}
          side="bottom"
          align="end"
          // Stop the content from taking focus from the input
          onOpenAutoFocus={e => e.preventDefault()}
          className="bg-popover max-h-[80vh] overflow-auto w-[75vw] max-w-[800px] p-4 rounded shadow-xl"
        >
          <div className="w-full overflow-hidden">
            {/* TODO: Investigate how to pass option groups */}
            {results && results.length > 0 && (
              <ol className="grid gap-2 overflow-hidden">
                {/* TODO: Replace with / create a `SearchableSelectGroup` */}
                {results.slice(0, 12).map((file, i) => {
                  return (
                    <GlobalSearchListItem
                      key={i}
                      repoId={repoId as string}
                      repoKind={repositoryKind as RepositoryKind}
                      file={file}
                      hidePopover={() => setPopoverIsShown(false)}
                    />
                  )
                })}
              </ol>
            )}
          </div>
        </SearchableSelectContent>
      </SearchableSelect>
    </div>
  )
}

export { GlobalSearch }
