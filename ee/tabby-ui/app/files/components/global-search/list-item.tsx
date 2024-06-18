'use client'

import React, {
  FormEventHandler,
  use,
  useContext,
  useEffect,
  useState
} from 'react'
import { darcula } from 'react-syntax-highlighter/dist/esm/styles/hljs'
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

import {
  encodeURIComponentIgnoringSlash,
  getProviderVariantFromKind
} from '../utils'
import { SourceCodeBrowserContext } from './source-code-browser'
import { resolveRepositoryInfoFromPath } from './utils'

interface GlobalSearchListItemProps {
  path: string
  repoKind: RepositoryKind
  repoId: string
}

export const GlobalSearchListItem = ({
  ...props
}: GlobalSearchListItemProps) => {
  const url = encodeURIComponentIgnoringSlash(
    `/repositories/${getProviderVariantFromKind(props.repoKind)}/${
      props.repoId
    }/resolve/${props.path}`
  )

  const { data, isLoading, error }: SWRResponse<ResolveEntriesResponse> =
    useSWRImmutable(url, fetcher)

  // const otherData = fetcher(url).then(data => {
  //   console.log('father', data)
  // })
  // console.log('otherData', otherData)

  // if (error) {
  //   // This is not always indicative of an error
  //   console.error(error)
  // }

  // if (!data) {
  //   return <div>Loading...</div>
  // }

  console.log('DATER', data)
  /**
   * TODO: We probably don't wanna fetch these individually;;; move to parent
   */

  return (
    <li>
      <div>TEST</div>
    </li>
  )
}
