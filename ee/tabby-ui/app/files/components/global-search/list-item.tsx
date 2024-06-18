'use client'

import React, {
  FormEventHandler,
  use,
  useContext,
  useEffect,
  useState
} from 'react'
import { darcula } from 'react-syntax-highlighter/dist/esm/styles/hljs'
import useSWRImmutable from 'swr/immutable'

import { graphql } from '@/lib/gql/generates'
import {
  GrepFile,
  GrepLine,
  GrepTextOrBase64,
  RepositoryKind
} from '@/lib/gql/generates/graphql'
import { client } from '@/lib/tabby/gql'
import { fetcher } from '@/lib/utils'

import { encodeURIComponentIgnoringSlash } from '../utils'
import { SourceCodeBrowserContext } from './source-code-browser'
import { resolveRepositoryInfoFromPath } from './utils'

interface GlobalSearchListItemProps {
  file: GrepFile & { keys: null }
  repo: string
}

export const GlobalSearchListItem = ({
  ...props
}: GlobalSearchListItemProps) => {
  const [path, setPath] = useState('')
  const [lines, setLines] = useState<GrepLine[]>([])

  useEffect(() => {
    setPath(props.file.path)
    setLines(props.file.lines)
  }, [props.file]) // Confirm this cleanup

  // WIP
  const url = encodeURIComponentIgnoringSlash(
    `/repositories/${props.repo}/resolve/${props.file.path}`
  )

  const fetch = useSWRImmutable(url, (url: string) => {
    debugger
    return fetcher(url, {
      responseFormatter: async response => {
        const blob = await response.blob()
        return blob
      }
    })
  })

  const { data, error } = fetch

  debugger

  if (error) {
    console.error(error)
  }

  if (!data) {
    return <div>Loading...</div>
  }

  console.log('DATER', data)
  /**
   * TODO: We probably don't wanna fetch these individually;;; move to parent
   */

  return (
    <li>
      <div>{path}</div>
      <ol>
        {lines.slice(0, 3).map((line, i) => (
          <li key={i}>nice</li>
        ))}
      </ol>
    </li>
  )
}
