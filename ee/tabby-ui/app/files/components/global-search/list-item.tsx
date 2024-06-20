'use client'

import React, { useEffect, useState } from 'react'
import Link from 'next/link'
import * as DropdownMenu from '@radix-ui/react-dropdown-menu'
import useSWRImmutable from 'swr/immutable'

import { GrepFile, GrepLine, RepositoryKind } from '@/lib/gql/generates/graphql'
import fetcher from '@/lib/tabby/fetcher'
import { Skeleton } from '@/components/ui/skeleton'
import { SearchableSelectOption } from '@/components/searchable-select'

import { SourceCodeBrowserContext } from '../source-code-browser'
import {
  encodeURIComponentIgnoringSlash,
  generateEntryPath,
  getProviderVariantFromKind
} from '../utils'
import { GlobalSearchSnippet } from './snippet'

interface GlobalSearchListItemProps {
  file: GrepFile
  repoKind: RepositoryKind
  repoId: string
  key: number
  hidePopover: () => void
}

export const GlobalSearchListItem = ({
  ...props
}: GlobalSearchListItemProps) => {
  const {
    activePath,
    currentFileRoutes,
    fileTreeData,
    activeRepo,
    activeRepoRef,
    repoMap,
    activeEntryInfo
  } = React.useContext(SourceCodeBrowserContext)

  const [blob, setBlob] = useState<Blob | undefined>(undefined)
  const [blobText, setBlobText] = useState<string | undefined>(undefined)

  const [lines, setLines] = useState<GrepLine[] | undefined>(undefined)

  // TODO: Convert to utility function
  const url = encodeURIComponentIgnoringSlash(
    `/repositories/${getProviderVariantFromKind(props.repoKind)}/${
      props.repoId
    }/resolve/${props.file.path}`
  )

  const { data } = useSWRImmutable(
    url,
    (url: string) =>
      fetcher(url, {
        responseFormatter: async response => {
          if (!response.ok) return undefined
          const blob = await response.blob()
          return blob
        }
      }),
    {
      onError() {
        // TODO: Add error handling
      }
    }
  )

  useEffect(() => {
    // TODO: remove?
    setBlob(data)

    const linesWithSubMatches = props.file.lines.filter(
      line => line.subMatches.length > 0
    )

    setLines(linesWithSubMatches)

    const blob2Text = async (blob: Blob) => {
      try {
        const b = await blob.text()
        setBlobText(b)
      } catch (e) {
        setBlobText(undefined)
      }
    }

    if (blob) {
      blob2Text(blob)
    }
  }, [props.file.lines, data, blob])

  return (
    <div>
      <Link
        href={`/files/${generateEntryPath(
          activeRepo,
          activeRepoRef?.name as string,
          props.file.path,
          'file'
        )}`}
        className="text-sm font-semibold mb-2"
      >
        {props.file.path}
      </Link>

      {lines && (
        <ol className="overflow-hidden grid gap-0.5">
          {lines.map((line, i) => {
            return (
              // TODO: Replace with /  `SearchableSelectItem`
              <GlobalSearchSnippet
                key={i}
                blobText={blobText as string}
                repoId={props.repoId}
                file={props.file}
                line={line}
                hidePopover={props.hidePopover}
              />
            )
          })}
        </ol>
      )}
    </div>
  )
}
