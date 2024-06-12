import React from 'react'

import { cn } from '@/lib/utils'
import { IconFileSearch } from '@/components/ui/icons'

import { SourceCodeBrowserContext } from './source-code-browser'
import { Errors } from './utils'

// import { Errors } from "./utils";

interface ErrorViewProps extends React.HTMLAttributes<HTMLDivElement> {
  error: Error | undefined
}

export const ErrorView: React.FC<ErrorViewProps> = ({ className, error }) => {
  const { activeEntryInfo, activeRepo, activeRepoRef } = React.useContext(
    SourceCodeBrowserContext
  )
  const basename = activeEntryInfo?.basename
  const isNotFound = error?.message === Errors.NOT_FOUND
  const isEmptyRepository = error?.message === Errors?.EMPTY_REPOSITORY
  // const isEmptyRepository = !!basename && (!activeRepo || !activeRepoRef?.name)

  let errorMessge = 'Not found'
  if (isEmptyRepository) {
    errorMessge = 'Empty repository'
  }

  return (
    <div
      className={cn('min-h-[30vh] flex items-center justify-center', className)}
    >
      <div className="flex flex-col items-center">
        <IconFileSearch className="mb-2 h-10 w-10" />
        <div className="font-semibold text-2xl">{errorMessge}</div>
      </div>
    </div>
  )
}
