'use client'

import React, { useEffect, useState } from 'react'

import { GrepFile, GrepLine, RepositoryKind } from '@/lib/gql/generates/graphql'

import { GlobalSearchListItem } from './list-item'

interface GlobalSearchResultsProps {
  results: GrepFile[] | null
  repoId?: string
  repositoryKind?: RepositoryKind
  hidePopover: () => void
}

export const GlobalSearchResults = ({ ...props }: GlobalSearchResultsProps) => {
  // Slice or batch load...

  //

  return (
    <>
      {props.results && props.results.length > 0 && (
        <ol className="grid gap-2 overflow-hidden">
          {/* TODO: Replace with / create a `SearchableSelectGroup` */}
          {props.results.slice(0, 12).map((file, i) => {
            return (
              <GlobalSearchListItem
                key={i}
                repoId={props.repoId as string}
                repoKind={props.repositoryKind as RepositoryKind}
                file={file}
                hidePopover={props.hidePopover}
              />
            )
          })}
        </ol>
      )}
    </>
  )
}
