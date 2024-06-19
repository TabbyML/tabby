'use client'

import React, {
  FormEventHandler,
  use,
  useContext,
  useEffect,
  useState
} from 'react'
import { lineNumbers } from '@codemirror/view'
import { toNumber } from 'lodash-es'
import { darcula } from 'react-syntax-highlighter/dist/esm/styles/hljs'
import { toast } from 'sonner'
import { SWRResponse } from 'swr'
import useSWRImmutable from 'swr/immutable'

import { graphql } from '@/lib/gql/generates'
import {
  GrepFile,
  GrepLine,
  GrepTextOrBase64,
  RepositoryKind
} from '@/lib/gql/generates/graphql'
import authEnhancedFetch from '@/lib/tabby/fetcher'
import fetcher from '@/lib/tabby/fetcher'
import { client } from '@/lib/tabby/gql'
import { ResolveEntriesResponse } from '@/lib/types'
import CodeEditor from '@/components/codemirror/codemirror'

import {
  encodeURIComponentIgnoringSlash,
  getProviderVariantFromKind
} from '../utils'
import { GlobalSearchSnippet } from './snippet'
import { SourceCodeBrowserContext } from './source-code-browser'
import { resolveRepositoryInfoFromPath } from './utils'

interface GlobalSearchListItemProps {
  file: GrepFile
  repoKind: RepositoryKind
  repoId: string
}

export const GlobalSearchListItem = ({
  ...props
}: GlobalSearchListItemProps) => {
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
    <li>
      <h5 className="text-sm font-semibold mb-2">{props.file.path}</h5>
      <ol className="overflow-hidden grid gap-0.5">
        {lines ? (
          lines.map((line, i) => {
            return (
              <GlobalSearchSnippet
                key={i}
                blobText={blobText as string}
                repoId={props.repoId}
                file={props.file}
                line={line}
              />
            )
          })
        ) : (
          <li>Loading...</li>
        )}
      </ol>
    </li>
  )
}
