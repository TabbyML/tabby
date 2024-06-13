import React from 'react'

import { cn } from '@/lib/utils'
import { IconFileSearch } from '@/components/ui/icons'

import { Errors } from './utils'

interface ErrorViewProps extends React.HTMLAttributes<HTMLDivElement> {
  error: Error | undefined
}

export const ErrorView: React.FC<ErrorViewProps> = ({ className, error }) => {
  const isEmptyRepository = error?.message === Errors?.EMPTY_REPOSITORY

  let errorMessge = 'Not found'
  if (isEmptyRepository) {
    errorMessge = 'Empty repository'
  }

  return (
    <div
      className={cn('flex min-h-[30vh] items-center justify-center', className)}
    >
      <div className="flex flex-col items-center">
        <IconFileSearch className="mb-2 h-10 w-10" />
        <div className="text-2xl font-semibold">{errorMessge}</div>
      </div>
    </div>
  )
}
